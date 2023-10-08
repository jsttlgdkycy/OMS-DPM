import torch
import numpy as np 
import argparse
import logging
import yaml
import sys
import time
import random
import os
import shutil

from dataset import get_predictor_dataloader
from optimizer import get_optimizer
from train import train
from test import test
from search import search

from modules.predictor import get_predictor, load_ckpt
from modules.noise_schedules import get_noise_schedule

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def get_logger(args):
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p')
    logger = logging.getLogger()
    os.makedirs(args.exp, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(args.exp, f"{args.type}.log"))
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    logger.info("Conducting Command: %s", " ".join(sys.argv))
    return logger

def main():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, default="predictor_cifar10_dpm_solver.yml", help="predictor configs")
    parser.add_argument("--gpu", type=str, default="0", help="device number")
    parser.add_argument("--type", type=str, default="train", help="1. train; 2. test; 3. search")
    parser.add_argument("--exp", type=str, default="code/predictor/exps/debug", help="dir to save the result of this experiment")
    parser.add_argument("--seed", type=int, default=3154, help="random seed")
    parser.add_argument("--resume", type=str, default=None, help="load checkpoint")
    parser.add_argument("--budget", type=int, default=None, help="The budget constraint of the search phase")
    
    args = parser.parse_args()
    args.config = os.path.join("code/predictor/configs", args.config)
    with open(args.config, "r", encoding="utf-8") as f: 
        config = yaml.safe_load(f)
    
    assert args.type in ["train", "test", "search"], "The xxx must be train or test or search!" 
    seed_everything(args.seed)
    
    # get logger
    logger = get_logger(args)
    
    # save config
    shutil.copyfile(args.config, os.path.join(args.exp, "predictor.yml"))
    
    # specify gpu
    if args.gpu and torch.cuda.is_available():
        args.gpu_flag = True
        device = torch.device('cuda')
        gpus = [int(d) for d in args.gpu.split(',')]
        args.gpu = gpus
        torch.cuda.set_device(gpus[0]) # currently only training & inference on single card is supported.
        logger.info("Using GPU(s). Available gpu count: {}".format(torch.cuda.device_count()))
    else:
        device = torch.device('cpu')
        logger.info("Using cpu!")
        
    # get noise schedule if needed
    if config["predictor"]["sampler_type"]=="dpm-solver" and config["predictor"]["timestep_type"]=="logSNR":
        noise_schedule = get_noise_schedule(config["predictor"])
    else:
        noise_schedule = None
    
    # get predictor
    print("Init predictor...")
    predictor = get_predictor(config["predictor"], noise_schedule).to(device)
    
    # get optimizer
    optimizer, scheduler = get_optimizer(config["training"]["optimizer"], predictor, config["training"]["epoch"])
    
    epoch = 0
    mean_loss = 0
    
    # load checkpoint
    if args.resume is not None:
        epoch = load_ckpt(args, device, predictor, optimizer)
    
    # get dataset
    if args.type in ["train", "test"]:
        print("Load dataset...")
        train_loader, valid_loader = get_predictor_dataloader(config, noise_schedule)
        # from dataset import get_predictor_dataloader_old
        # train_loader, valid_loader = get_predictor_dataloader_old(old_config)
        
    if args.type=="train":
        train(config, epoch, mean_loss, predictor, train_loader, valid_loader, optimizer, scheduler, args.exp, logger)
    elif args.type=="test":
        test(config, predictor, valid_loader, logger)
    elif args.type=="search":
        search(args, config, predictor, logger)
    else:
        raise NotImplementedError("The xxx must be train or test or search!")
    

if __name__=="__main__":
    main()