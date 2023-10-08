import torch
from scipy.stats import kendalltau

def cal_kendall_tau(set1, set2):
    corr = kendalltau(set1, set2).correlation
    return corr

def valid_one_epoch(predictor, valid_loader, logger, set="valid set"):
    predictor.eval()
    with torch.no_grad():
        gt_scores = []
        predict_scores = []
        for ms, gt_score in valid_loader:
            predict_score = predictor(ms)
            for i in range(len(predict_score)):
                gt_scores.append(gt_score[i].item())
                predict_scores.append(predict_score[i].item())
    kd = cal_kendall_tau(gt_scores, predict_scores)
    logger.info(f"The KD of predict scores and gt scores on {set} is {kd}")
    return kd
    
def test(predictor, valid_loader, logger):
    valid_one_epoch(predictor, valid_loader, logger)