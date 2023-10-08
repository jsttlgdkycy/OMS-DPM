import numpy as np
import torch

def compare_data(x, y, compare_threshold, max_compare_ratio):
    for key in x.keys():
        x[key] = x[key].detach().cpu().numpy()
        bs = x[key].shape[0]
    gt_score = y.detach().cpu().numpy()
    diff = gt_score[:, None] - gt_score
    abs_diff = np.triu(np.abs(diff), 1)
    ex_thresh_inds = np.where(abs_diff > compare_threshold)
    ex_thresh_num = len(ex_thresh_inds[0])
    n_max_pairs = int(max_compare_ratio * bs)
    if ex_thresh_num > n_max_pairs:
        keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
        ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])

    ms_1, ms_2 = {}, {}
    better_lst = (diff > 0)[ex_thresh_inds]   
    for key in x.keys():
        ms_1[key], ms_2[key] = torch.tensor(x[key][ex_thresh_inds[1]]), torch.tensor(x[key][ex_thresh_inds[0]]),    

    return ms_1, ms_2, better_lst
        
def get_timesteps_for_dpm_solver(order_sequences, timestep_type, max_timesteps, noise_schedule):
    
    t_continuous = torch.zeros(order_sequences.shape[0], order_sequences.shape[1]+1).to(torch.float32).to(order_sequences.device)
    
    for batch_index, order_sequence in enumerate(order_sequences):
        timestep_nums = (order_sequence!=0).sum()
        timesteps = torch.zeros(order_sequences.shape[0], max_timesteps + 1)
        
        t_T = 1.000
        t_0 = 1e-4 if order_sequence.sum()>=15 else 1e-3
        device = order_sequence.device

        if timestep_type=="logSNR":
            lambda_T = noise_schedule.marginal_lambda(torch.tensor(t_T).to("cpu")).to("cpu")
            lambda_0 = noise_schedule.marginal_lambda(torch.tensor(t_0).to("cpu")).to("cpu")
            timesteps = torch.linspace(lambda_0, lambda_T, timestep_nums + 1)
            t_continuous[batch_index, :timestep_nums+1] = noise_schedule.inverse_lambda(timesteps)[:timestep_nums+1].to(device)
        elif timestep_type=="time_uniform":
            linear_steps = torch.linspace(t_0, t_T, timestep_nums + 1)
            t_continuous[batch_index, :timestep_nums+1] = linear_steps[:timestep_nums+1].to(device)
        else:
            raise NotImplementedError(f"{timestep_type} is not supported currently!")

    t_discrete = 1000. * torch.max(t_continuous - 1. / 1000, torch.zeros_like(t_continuous).to(device))
    
    return t_discrete

def gather_active_solver(ms):
    
    new_ms = ms.reshape(-1, 3)
    active_index = torch.where(new_ms.sum(dim=1)!=0)[0]
    inactive_index = torch.where(new_ms.sum(dim=1)==0)[0]
    active_solver = new_ms[active_index].reshape(-1)
    inactive_solver = new_ms[inactive_index].reshape(-1)
    return torch.cat([active_solver, inactive_solver],dim=0)

def reorder(ms):
    
    if len(ms.shape)==1:
        return gather_active_solver(ms)    
    elif len(ms.shape)==2:
        new_ms = torch.zeros_like(ms)
        for b in range(ms.shape[0]):
            new_ms[b] = gather_active_solver(ms[b])
    else:
        raise ValueError("The shape of model schedule should be [bs, L] or [L]!")
    
    return new_ms

def get_timesteps_for_ddim(length, max_timesteps=1000):
    
    timesteps = torch.arange(0, max_timesteps, max_timesteps // length)
    
    return timesteps
