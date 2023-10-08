import os
import torch
import random
import numpy as np
import copy

class model_schedule:
    def __init__(self, config):
        self.model_zoo_size = config["predictor"]["model_embedder"]["model_zoo_size"] + 1 # +1 for the null model
        self.max_length = config["search"]["max_length"]
        self.mutate_prob = config["search"]["mutate_prob"]
        self.init_correct_prob = config["search"]["init_correct_prob"]
        self.step_size = config["search"]["step_size"]
        
        # load model zoo latency
        self.model_zoo_latency = torch.load(config["search"]["model_zoo_latency"])
        if self.model_zoo_latency[0]!=0: # null model
            self.model_zoo_latency.insert(0, 0)
        self.model_zoo_latency = np.array(self.model_zoo_latency)
            
        self.cost = None
        self.perf = None
        self.ms = None # numpy
        
    def set_ms(self, ms):
        self.ms = ms
        
    def random_init_ms(self):
        pass
    
    def get_perf(self, predictor):
        data = {
            "ms":torch.from_numpy(self.ms).to(next(predictor.parameters()).device).unsqueeze(0)
        }
        if self.perf is None:
            with torch.no_grad():
                self.perf = predictor(data).item()
        return self.perf
                
    def get_cost(self):
        if self.cost is None:
            self.cost = np.sum(self.model_zoo_latency[self.ms]).item()
        return self.cost
    
    def mutate(self):
        '''
        Return a new model schedule instance while keeping the current model schedule instance unchanged
        '''
        pass
    
    def correct_cost(self, tolerance, max_init_time, budget):
        init_time = 0
        while(1):
            if self.get_cost()>budget:
                self.mutate(mode="lighter", return_type="self", mutate_prob=self.init_correct_prob)
            elif self.get_cost()<tolerance * budget:
                self.mutate(mode="heavier", return_type="self", mutate_prob=self.init_correct_prob)
            else:
                print(f"Finish correcting.")
                break
            init_time += 1
            if init_time>max_init_time:
                print(f"Faile to initialize a model schedule whose cost is in [{tolerance}*{budget}, {budget}]. The cost of initial model schedule is {self.get_cost()}")
                import ipdb; ipdb.set_trace()
                break
    
    def check(self):
        '''
        Check whether the format of model schedule is correct
        '''
        pass
    
class dpmsolver_model_schedule(model_schedule):
    def __init__(self, config):
        super().__init__(config)
        
    def random_init_ms(self, tolerance=0.9, max_init_time=200, budget=None):
        ms = []
        K = self.max_length // 3
        for i in range(K):
            solver_order = random.randint(0, 3)
            for j in range(0, 3):
                if j<solver_order:
                    ms.append(random.randint(1, self.model_zoo_size-1))
                else:
                    ms.append(0)
        ms = np.array(ms)
        self.set_ms(ms)
        
        if budget is not None:
            self.correct_cost(tolerance, max_init_time, budget)
        print(f"Finish initialization. The cost of initial model schedule is {self.get_cost()}")
            
    def mutate(self, mode="normal", return_type="new", mutate_prob=None):
        assert mode in ["normal", "heavier", "lighter"]
        assert return_type in ["new", "self"]
        
        if mutate_prob is None:
            mutate_prob = self.mutate_prob
        
        if return_type=="new":
            new_ms = copy.deepcopy(self)
            prev_ms = self.ms
        else:
            new_ms = self
            prev_ms = copy.deepcopy(self.ms)
            
        random_step_size = random.randint(1, self.step_size)
        indices = torch.tensor(random.sample(range(0, len(new_ms.ms)), random_step_size))
        for index in indices:
            if index % 3!=2 and new_ms.ms[index+1]!=0:
                candidate_list = np.array(range(1, self.model_zoo_size))
            else:
                candidate_list = np.array(range(self.model_zoo_size))
            if mode=="heavier":
                candidate_list = candidate_list[np.where(new_ms.model_zoo_latency[candidate_list]>=new_ms.model_zoo_latency[new_ms.ms[index]])[0]]
            elif mode=="lighter":
                candidate_list = candidate_list[np.where(new_ms.model_zoo_latency[candidate_list]<=new_ms.model_zoo_latency[new_ms.ms[index]])[0]]
            if index % 3==0 or new_ms.ms[index-1]!=0:
                if random.uniform(0, 1)<mutate_prob:
                    new_ms.ms[index] = random.choice(candidate_list)
                    
        if not (new_ms.ms==prev_ms).all():
            new_ms.cost = None
            new_ms.perf = None
                    
        if return_type=="new":
            return new_ms
    
    def check(self):
        if len(self.ms)!=self.max_length:
            return False
        for i in range(self.max_length):
            if self.ms[i]==0 and self.ms[i+1]!=0 and i % 3!=2:
                return False
        return True
    
class ddim_model_schedule(model_schedule):
    def __init__(self, config):
        super().__init__(config)
        
    def random_init_ms(self, tolerance=0.9, max_init_time=200, budget=None):
        ms = np.random.randint(low=0, high=self.model_zoo_size, size=self.max_length)
        self.set_ms(ms)
        
        if budget is not None:
            self.correct_cost(tolerance, max_init_time, budget)
        print(f"Finish initialization. The cost of initial model schedule is {self.get_cost()}")
        
    def mutate(self, mode="normal", return_type="new", mutate_prob=None):
        assert mode in ["normal", "heavier", "lighter"]
        assert return_type in ["new", "self"]
        
        if mutate_prob is None:
            mutate_prob = self.mutate_prob
        
        if return_type=="new":
            new_ms = copy.deepcopy(self)
            prev_ms = self.ms
        else:
            new_ms = self
            prev_ms = copy.deepcopy(self.ms)
        
        random_step_size = random.randint(1, self.step_size)
        indices = torch.tensor(random.sample(range(0, len(new_ms.ms)), random_step_size))
        for index in indices:
            candidate_list = np.array(range(self.model_zoo_size))
            if mode=="heavier":
                candidate_list = candidate_list[np.where(new_ms.model_zoo_latency[candidate_list]>=new_ms.model_zoo_latency[new_ms.ms[index]])[0]]
            elif mode=="lighter":
                candidate_list = candidate_list[np.where(new_ms.model_zoo_latency[candidate_list]<=new_ms.model_zoo_latency[new_ms.ms[index]])[0]]
            if random.uniform(0, 1)<mutate_prob:
                new_ms.ms[index] = random.choice(candidate_list)
        if not (new_ms.ms==prev_ms).all():
            new_ms.cost = None
            new_ms.perf = None
        
        if return_type=="new":
            return new_ms
    
    def check(self):
        return True
    
class controller:
    def __init__(self, config, predictor, logger):
        # search configurations
        self.config = config
        self.sampler_type = config["predictor"]["sampler_type"]
        self.smaller_score = 1 if config["search"]["smaller_score"] else -1
        self.max_init_time = config["search"]["max_init_time"]
        self.max_num_next_generation = config["search"]["max_num_next_generation"]
        self.max_mutate_time_one_iter = config["search"]["max_mutate_time_one_iter"]
        self.init_tolerance = config["search"]["init_tolerance"]
        self.max_candidate_parents = config["search"]["max_candidate_parents"]
        self.max_population_size = config["search"]["max_population_size"]
        self.epoch = config["search"]["epoch"]
        self.log_every = config["search"]["log_every"]
        
        # predictor
        self.predictor = predictor
        
        # logger
        self.logger = logger
    
    def get_initial_ms(self, budget, init_ms=None):
        if self.sampler_type=="dpm-solver":
            ms = dpmsolver_model_schedule(self.config)
        elif self.sampler_type=="ddim":
            ms = ddim_model_schedule(self.config)
        else:
            raise NotImplementedError(f"Sampler type {self.sampler_type} is not supported!")
        
        if init_ms is not None:
            ms.set_ms(init_ms)
        else:
            ms.random_init_ms(tolerance=self.init_tolerance, max_init_time=self.max_init_time, budget=budget)
            
        return ms

    def step(self, budget):
        '''
        Generate the next generation of model schedules.
        '''
        next_generation = []
        mutate_time = 0
        while(1):
            # get candidates
            indices = random.sample(range(0, len(self.population)), self.max_candidate_parents) if len(self.population)>self.max_candidate_parents else range(0, len(self.population))
            
            # get parent
            best_index = min(indices, key = lambda x: self.smaller_score * self.population[x].get_perf(self.predictor))
            parent = self.population[best_index]
            
            # mutate
            new_ms = parent.mutate()
            
            if new_ms.get_cost()<=budget: # abandon model schedules with latencies exceeding the budget
                next_generation.append(new_ms)
            
            mutate_time += 1
            
            if len(next_generation)==self.max_num_next_generation or mutate_time==self.max_mutate_time_one_iter:
                break
            
        return next_generation
    
    def save_population(self, save_path):
        clean_ms_population = []
        for i in range(len(self.population)):
            clean_ms_population.append(self.population[i].ms.tolist())
        torch.save(clean_ms_population, save_path)
            
    def search(self, budget, save_path):
        initial_ms = self.get_initial_ms(budget)
        
        self.population = [initial_ms]
        
        for i in range(self.epoch):
            # generate the next generation
            next_generation = self.step(budget)
            self.population += next_generation
            
            # eliminate the individuals with poor performance, keeping the size of population smaller than a particular value
            self.population.sort(key=lambda x: self.smaller_score * x.get_perf(self.predictor))
            if len(self.population)>self.max_population_size:
                self.population = self.population[:self.max_population_size]

            # print info
            best_score = self.population[0].get_perf(self.predictor)
            if i % self.log_every==1:
                self.logger.info(f"Epoch {i} | best predict score {best_score}")
        
        self.save_population(os.path.join(save_path, "final_population.pth"))