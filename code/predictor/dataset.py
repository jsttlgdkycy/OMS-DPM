import torch
import torch.utils.data as data
from pathlib import Path
import random
import os

from modules.predictor_utils import reorder, get_timesteps_for_dpm_solver

class ms_perf_dataset(data.Dataset):
    
    def __init__(self, data, noise_schedule=None, need_timesteps=False):
        super().__init__()
        self.data = data
        self.need_timesteps = need_timesteps
        
        self.ms = []
        self.score = []
        self.timesteps = []
        for d in self.data:
            try:
                data = torch.load(d)
            except:
                data = d
            if isinstance(data, dict):
                ms = torch.tensor(data["model_schedule"]) if not torch.is_tensor(data["model_schedule"]) else data["model_schedule"]
                score = torch.tensor(data["score"]) if not torch.is_tensor(data["score"]) else data["score"]
            elif isinstance(data, list): 
                ms = torch.tensor(data[0]) if not torch.is_tensor(data[0]) else data[0]
                score = torch.tensor(data[1]) if not torch.is_tensor(data[1]) else data[1]
            if self.need_timesteps:
                ms = reorder(ms)
                order_sequences = (ms!=0).reshape(len(ms) // 3, -1).sum(dim=1).unsqueeze(0)
                timesteps = get_timesteps_for_dpm_solver(order_sequences, "logSNR", len(ms), noise_schedule).squeeze()
            else:
                timesteps = None
            self.ms.append(ms)
            self.score.append(score)
            self.timesteps.append(timesteps)
        
    def __len__(self):
        return len(self.ms)
    
    def __getitem__(self, index):
        ms = self.ms[index]
        score = self.score[index]
        data_dict = {"ms": ms, "reorder":False}
        if self.need_timesteps:
            data_dict["timesteps"] = self.timesteps[index]

        return data_dict, score

def get_predictor_dataset(config, noise_schedule=None, ext="pth"):
    base_path = config["dataset"]["path"]
    if os.path.isdir(base_path):
        data_path = [p for p in Path(f'{base_path}').glob(f'**/*.{ext}')]
        
        train_length = int(len(data_path) * config["dataset"]["train_ratio"])
        train_index = random.sample(list(range(len(data_path))), train_length)

        train_data_path = []
        valid_data_path = []
        for i in range(len(data_path)):
            if i in train_index:
                train_data_path.append(data_path[i])
            else:
                valid_data_path.append(data_path[i])
        
        print(f"The size of all set is {len(data_path)}")
        print(f"The size of training set is {len(train_data_path)}")

        need_timesteps = config["predictor"]["sampler_type"]=="dpm-solver"
        train_set = ms_perf_dataset(train_data_path, noise_schedule, need_timesteps)
        valid_set = ms_perf_dataset(valid_data_path, noise_schedule, need_timesteps)
        
    elif os.path.isfile(base_path):
        all_data = torch.load(base_path)
        
        train_length = int(len(all_data) * config["dataset"]["train_ratio"])
        train_index = random.sample(list(range(len(all_data))), train_length)

        train_data = []
        valid_data = []
        for i in range(len(all_data)):
            if i in train_index:
                train_data.append(all_data[i])
            else:
                valid_data.append(all_data[i])
        
        print(f"The size of all set is {len(all_data)}")
        print(f"The size of training set is {len(train_data)}")

        need_timesteps = config["predictor"]["sampler_type"]=="dpm-solver"
        train_set = ms_perf_dataset(train_data, noise_schedule, need_timesteps)
        valid_set = ms_perf_dataset(valid_data, noise_schedule, need_timesteps)
        
    return train_set, valid_set

def get_predictor_dataloader(config, noise_schedule=None):
    trainset, validset = get_predictor_dataset(config, noise_schedule)
    train_loader = data.DataLoader(
            trainset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"].get("num_workers", 0),
        )
    valid_loader = data.DataLoader(
            validset,
            batch_size=config["testing"]["batch_size"],
            shuffle=False,
            num_workers=config["testing"].get("num_workers", 0),
        )
    
    return train_loader, valid_loader