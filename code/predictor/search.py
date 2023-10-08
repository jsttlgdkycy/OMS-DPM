import torch
import random
import numpy

from search_utils import *

def search(args, config, predictor, logger):
    assert args.budget is not None, "need to specify a budget to constrain the search"
    
    predictor.eval()
    
    evo_controller = controller(config, predictor, logger)
    evo_controller.search(args.budget, args.exp)
    
    print("Search complete")