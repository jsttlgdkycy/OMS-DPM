import random
import torch
import numpy as np

def get_dirichlet(alpha, n):
    alphas = [alpha] * n
    samples = np.random.dirichlet(alphas, 1).squeeze(0)
    return samples

def get_multinomial(num_samples, prob_vector):
    ms = np.random.choice(len(prob_vector), num_samples, p=prob_vector) + 1
    return ms

def correct_for_dpm_solver(ms):
    K = len(ms) // 3
    for i in range(K):
        ms[i*3:(i+1)*3] = np.concatenate((ms[i*3:(i+1)*3][ms[i*3:(i+1)*3] != 0], ms[i*3:(i+1)*3][ms[i*3:(i+1)*3] == 0]))
            
def randomly_set_null(ms, prob):
    for i in reversed(range(len(ms))):
        if random.uniform(0, 1)<prob:
            ms[i] = 0
    # TODO: Add more sampling strategies. For example, set different probabilities of timesteps near the noise-end and image-end are set to 0 for DDIM; or set different probabilities for different solver orders for DPM-Solver.
    # TODO: Set up a randomly drawing process from different sampling strategies. 

def get_model_schedule(config, ms_length=100, solver_type=None, model_zoo_size=None):
    if config.type=="specify":
        return config.specify.ms
    elif config.type=="load":
        data = torch.load(config.load.load_path)
        if isinstance(data, list):
            if not isinstance(data[0], int):
                return data[config.load.rank]["ms"].tolist()
            else:
                return data 
        else:
            assert isinstance(data, [np.ndarray, torch.Tensor])
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            return data.tolist()
    elif config.type=="multinomial":
        prob_vector = config.multinomial.prob_vector
        if len(prob_vector)!=model_zoo_size:
            raise ValueError(f"The length of prob vector {len(prob_vector)} does not align with the size of model zoo {model_zoo_size}!")
        ms = get_multinomial(ms_length, prob_vector)
        randomly_set_null(ms, config.multinomial.set_zero_prob)
        if solver_type=="dpmsolver":
            correct_for_dpm_solver(ms)
        ms = ms.tolist()
    elif config.type=="multinomial+hierarchical":
        alpha = config.hierarchical.alpha
        prob_vector = get_dirichlet(alpha, model_zoo_size)
        ms = get_multinomial(ms_length, prob_vector)
        randomly_set_null(ms, config.multinomial.set_zero_prob)
        if solver_type=="dpmsolver":
            correct_for_dpm_solver(ms)
        ms = ms.tolist()
    else:
        raise NotImplementedError(f"Method of getting model schedule \"{config.type}\" is not supported!")
    
    print(f"The sampled model schedule is {ms}") 
    return ms