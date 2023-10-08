import torch
import torch.nn as nn
import importlib

from modules.model_embedder import *
from modules.ms_encoder import *
from modules.regression_head import *
from modules.timestep_encoder import *
from modules.predictor_utils import *
from modules.noise_schedules import *

def get_predictor(config, noise_schedule=None):
    sampler_type = config["sampler_type"]
    if sampler_type=="dpm-solver":
        predictor = ms_predictor_dpm_solver(config, noise_schedule)
    elif sampler_type=="ddim":
        predictor = ms_predictor_ddim(config)
    else:
        raise NotImplementedError(f"Sampler type {sampler_type} is not supported")
    return predictor

def load_ckpt(args, device, predictor, optimizer):
    state_dict = torch.load(args.resume, map_location=device)
    if isinstance(state_dict, dict):
        predictor.load_state_dict(state_dict["predictor"])
        optimizer.load_state_dict(state_dict["optimizer"])  
        epoch = state_dict["epoch"]
    elif isinstance(state_dict, list):
        predictor.load_state_dict(state_dict[0])
        optimizer.load_state_dict(state_dict[1])  
        epoch = state_dict[2]
    return epoch

class ms_predictor(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # model embedder
        self.model_embedder = model_embedder(config["model_embedder"])
        
        # timestep encoder
        self.timestep_encoder = timestep_encoder(config["timestep_encoder"])
        
        # model sequence encoder (LSTM)
        # Since the dimensions of the model embedder of the predictor of two samplers are not the same, it is left to be defined in the corresponding subclasses
        
        # score regression head
        input_size = config["ms_encoder"]["hidden_size"]
        self.regression_head = regression_head(config["regression_head"], input_size)
        
        # loss 
        self.loss_type = config["loss"]["loss_type"]
        self.compare_threshold = config["loss"]["ranking"]["compare_threshold"]
        self.max_compare_ratio = config["loss"]["ranking"]["max_compare_ratio"]
    
    def forward(self):
        pass
    
    def cal_loss(self, data, gt_score):
        if self.loss_type=="ranking":
            loss = self.update_compare(data, gt_score)
        elif self.loss_type=="mse":
            loss = self.update_predict(data, gt_score)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} is not supported currently!")
        return loss
        
    def update_compare(self, data, gt_score):
        data_1, data_2, better_lst = compare_data(data, gt_score, self.compare_threshold, self.max_compare_ratio)
        s_1 = self.forward(data_1)
        s_2 = self.forward(data_2)
        better_pm = 2 * s_1.new(np.array(better_lst, dtype=np.float32)) - 1
        zero_ = s_1.new([0.])
        margin = self.config["loss"]["ranking"]["compare_margin"]
        margin = s_1.new([margin])
        pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        return pair_loss
    
    def update_predict(self, data, gt_score):
        pred_score = self.forward(data)
        return (pred_score - gt_score).square().mean()
    
class ms_predictor_dpm_solver(ms_predictor): 
    # Currently only support for single-step DPM-Solver. 
    # TODO: multi-step DPM-Solver predictor
    
    def __init__(self, config, noise_schedule=None):
        super().__init__(config)
        
        # get solver encoder
        self.solver_encoder = nn.Sequential()
        in_dim = config["model_embedder"]["embedding_dim"] * 3
        for i in range(len(config["solver_encoder"]["out_dims"])):
            self.solver_encoder.add_module(f"solver_encoder_fc_{i}", nn.Linear(in_dim, config["solver_encoder"]["out_dims"][i]))
            in_dim = config["solver_encoder"]["out_dims"][i]
            if i!=len(config["solver_encoder"]["out_dims"])-1:
                self.solver_encoder.add_module(f"active_{i}", nn.Sigmoid())
        self.order_embedder = nn.Embedding(4, config["solver_encoder"]["order_emb_dim"])
                
        # model sequence encoder (LSTM)
        if config["timestep_encoder"]["shift"]:
            input_size = 2 * config["timestep_encoder"]["output_temb_dim"] + in_dim + config["solver_encoder"]["order_emb_dim"]
        else:
            input_size = config["timestep_encoder"]["output_temb_dim"] + in_dim + config["solver_encoder"]["order_emb_dim"]
        self.ms_encoder = ms_encoder(config["ms_encoder"], input_size)
                
        # get timestep split type for timestep compute
        self.timestep_type = config["timestep_type"]
        
        # get noise schedule if needed
        if self.timestep_type=="logSNR" and noise_schedule is None:
            noise_scheduler_type = config["noise_schedule"]["name"]
            module = importlib.import_module("modules.noise_schedules")
            if hasattr(module, noise_scheduler_type):
                ns_class = getattr(module, noise_scheduler_type)
                if config["noise_schedule"]["schedule"]=="discrete": # need to compute beta to instantiate a NS class
                    linear_start = config["noise_schedule"]["beta_start"]
                    linear_end = config["noise_schedule"]["beta_end"]
                    betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, 1000, dtype=torch.float64) ** 2
                    ns_init_config = {
                        "schedule":config["noise_schedule"]["schedule"],
                        "betas":betas,
                    }
                else:
                    ns_init_config = {
                        "schedule":config["noise_schedule"]["schedule"],
                    }
                self.noise_schedule = ns_class(**ns_init_config)
            else:
                raise NotImplementedError(f"{noise_scheduler_type} is not defined! Check the config")
        else:
            self.noise_schedule = noise_schedule
        
        
    def forward(self, data):
        '''
        Args:
            ms: dpm-solver model schedule. shape: [bs, L] (see eq.6 in our paper https://arxiv.org/abs/2306.08860)
            timesteps: Pre-computing the timestep before training will save time of timestep computing in the forward phase
        '''
        
        # get input data and set device
        ms = data["ms"].to(list(self.parameters())[0].device)
        
        timesteps = data.get("timesteps", None)
        if timesteps is not None:
            timesteps = timesteps.to(ms.device)
            
        # reorder model schedule to gather all active solver to the front
        need_reorder = data.get("reorder", [True])
        if need_reorder[0]:
            ms = reorder(ms)

        # get the number of solvers of the input model schedule
        K = ms.shape[1] // 3
        
        # init a matrix to put the order of each solver
        order_sequence = torch.zeros([ms.shape[0], K], dtype=torch.int64).to(ms.device) 
        # get solver embedding sequence
        temp_emb = self.model_embedder(ms) # [bs, L, dim]
        solver_sequence = temp_emb.reshape(temp_emb.shape[0], K, -1)
        order_sequence = (ms!=0).reshape(ms.shape[0], K, -1).sum(dim=2)
        solver_emb = self.solver_encoder(solver_sequence)
        order_emb = self.order_embedder(order_sequence)            
        solver_emb = torch.cat([solver_emb, order_emb], dim=2)        
            
        # get timesteps if needed
        if timesteps is None:
            timesteps = get_timesteps_for_dpm_solver(order_sequence, self.timestep_type, K, self.noise_schedule)

        # get timestep embeddings
        timestep_emb = self.timestep_encoder(timesteps)
        if self.config["timestep_encoder"]["shift"]:
            timestep_emb = torch.cat([timestep_emb[:, 1:, :], timestep_emb[:, :-1, :]], dim=2)
        else:
            timestep_emb = timestep_emb[:, :-1, :]

        # mask the inactivate solver
        timestep_emb *= (order_sequence!=0).unsqueeze(-1)

        # concat solver embedding and timestep embedding
        whole_emb = torch.cat([solver_emb, timestep_emb], dim=2)
        
        # get overall model schedule embedding
        ms_emb = self.ms_encoder(whole_emb)
        
        # regress the final score
        score = self.regression_head(ms_emb)
        
        return score

class ms_predictor_ddim(ms_predictor):
    def __init__(self, config):
        super().__init__(config)
        self.max_timesteps = config["max_timesteps"]
        
        # model sequence encoder (LSTM)
        input_size = config["timestep_encoder"]["output_temb_dim"] + config["model_embedder"]["embedding_dim"]
        self.ms_encoder = ms_encoder(config["ms_encoder"], input_size)
        
    def forward(self, data):
        '''
        Args:
            ms: ddim model schedule. shape: [bs, L] (see eq.6 in our original paper)
        '''
        # get input data and set device
        ms = data["ms"].to(list(self.parameters())[0].device)
        
        # get model embedding sequence
        model_emb = self.model_embedder(ms)
        
        # get timesteps
        timesteps = get_timesteps_for_ddim(ms.shape[1], self.max_timesteps).to(ms.device)
        
        # get timestep embeddings
        timestep_emb = self.timestep_encoder(timesteps)
        if len(timestep_emb.shape)==2:
            timestep_emb = timestep_emb.unsqueeze(0).repeat(ms.size(0), 1, 1) 
        
        # concat model embeddings and timestep embedding
        whole_emb = torch.cat([model_emb, timestep_emb], dim=2)
        
        # get overall model schedule embedding
        ms_emb = self.ms_encoder(whole_emb)
        
        # regress the final score
        score = self.regression_head(ms_emb)
        
        return score
        