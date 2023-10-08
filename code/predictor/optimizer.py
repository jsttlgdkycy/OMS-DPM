import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def get_optimizer(config, predictor, num_epochs):
    lr = config["learning_rate"]
    wd = config["weight_decay"]
    eta_min = config["eta_min"]
    if config["type"]=='sgd':
        optimizer = optim.SGD(predictor.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    elif config["type"] == 'adam':
        optimizer = optim.Adam(predictor.parameters(), lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    scheduler = None
    
    return optimizer, scheduler 