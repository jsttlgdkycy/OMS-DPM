import torch
import os

from test import valid_one_epoch

def save_checkpoint(predictor, optizmier, epoch, exp_path, name=None):
    data = {
        "predictor":predictor.state_dict(),
        "optimizer":optizmier.state_dict(),
        "epoch":epoch,
    }
    if name is None:
        name = f"checkpoint_{epoch}.pth"
    else:
        name = f"{name}.pth"
    torch.save(data, os.path.join(exp_path, name))
    print(f"Save data to {os.path.join(exp_path, name)}")

def train_one_epoch(predictor, train_loader, optimizer):
    predictor.train()
    epoch_mean_loss = 0
    for step, (data, gt_score) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = predictor.cal_loss(data, gt_score)
        loss.backward()
        optimizer.step()
        epoch_mean_loss = (epoch_mean_loss * step + loss.item()) / (step + 1)
    return epoch_mean_loss

def train(config, epoch, mean_loss, predictor, train_loader, valid_loader, optimizer, scheduler, exp_path, logger):
    end_epoch = config["training"]["epoch"]
    for i in range(epoch, end_epoch):
        epoch_mean_loss = train_one_epoch(predictor, train_loader, optimizer)
        mean_loss = (i * mean_loss + epoch_mean_loss) / (i + 1)
        logger.info(f"Epoch {i} | Loss: {epoch_mean_loss} | Average loss: {mean_loss}")
        if i % config["training"]["save_every"] == 0:
            save_checkpoint(predictor, optimizer, i, exp_path)
        if i % config["training"]["valid_every"] and i!=0:
            valid_one_epoch(predictor, train_loader, logger, set="train set")
            valid_one_epoch(predictor, valid_loader, logger, set="valid set")
        # scheduler.step()
            
    save_checkpoint(predictor, optimizer, end_epoch-1, exp_path, name="final")

        