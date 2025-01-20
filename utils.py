import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import torch
import torch.nn.functional as F
import pandas as pd
def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic

#输出各个性能指标
def performance_fit(y_label, y_output):
    y_output_logistic = fit_function(y_label, y_output)

    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output_logistic-y_label) ** 2).mean())

    0.863

    return PLCC, SRCC, KRCC, RMSE


def performance_no_fit(y_label, y_output):
    PLCC = stats.pearsonr(y_output, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_label-y_label) ** 2).mean())

    return PLCC, SRCC, KRCC, RMSE


class L1Loss(torch.nn.Module):

    def __init__(self, **kwargs):
        super(L1Loss, self).__init__()
        self.l1_w = kwargs.get("l1_w", 1)

    def forward(self, preds, gts):
        preds = preds.view(-1)
        gts = gts.view(-1)
        l1_loss = F.l1_loss(preds, gts) * self.l1_w

        return l1_loss

class L1RankLoss(torch.nn.Module):
    """
    L1 loss + Rank loss
    """

    def __init__(self, **kwargs):
        super(L1RankLoss, self).__init__()
        self.l1_w = kwargs.get("l1_w", 1)  #将l1_w的值赋值给self.l1_w，如果不存在l1_w则赋值为1
        self.rank_w = kwargs.get("rank_w", 1) #原论文=1
        self.hard_thred = kwargs.get("hard_thred", 1)
        self.use_margin = kwargs.get("use_margin", False)

    def forward(self, preds, gts):
        preds = preds.view(-1)  #将preds变成一个一维张量，具体大小根据系统自动推断
        gts = gts.view(-1)
        # l1 loss  F.l1_loss 函数来计算预测值 preds 与目标值 gts 之间的 L1 损失，并将结果乘以 self.l1_w。这样可以根据 self.l1_w 来调整 L1 损失的权重。
        l1_loss = F.l1_loss(preds, gts) * self.l1_w

        # simple rank  文中是计算两个视频之间，而代码中是计算自己和自己的转置
        n = len(preds)
        preds = preds.unsqueeze(0).repeat(n, 1)  #preds.unsqueeze(0) 的作用是在 preds 这个张量的第 0 维（即最外层的维度）上增加一个维度，将原本的一维张量转变为二维张量。然后使用 repeat(n, 1) 方法将其沿着指定维度复制 n 次，从而得到一个形状为 (n, *) 的新张量，其中 * 表示 preds 张量原本的大小。
        preds_t = preds.t()  #t()将矩阵转置
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = torch.sign(img_label - img_label_t)  #sign符号函数
        masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (torch.abs(img_label - img_label_t) > 0)
        if self.use_margin:
           # rank_loss = masks_hard * torch.relu(torch.abs(torch.abs(img_label - img_label_t) - masks * (preds - preds_t)))
            rank_loss = masks_hard * torch.relu(torch.abs(img_label - img_label_t) - masks * (preds - preds_t))
        else:
           # rank_loss=masks_hard*torch.relu(torch.abs(torch.abs(img_label - img_label_t) - masks * (preds - preds_t)))
            rank_loss = masks_hard * torch.relu(- masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)
        loss_total = l1_loss + rank_loss * self.rank_w
        return loss_total