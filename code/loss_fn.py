import torch
import torch.nn as nn

class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.65, size_average=True):
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, logits, targets):
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.where(t >= 0.5, l, 1 - l)
        a = torch.where(t >= 0.5, self.alpha, 1 - self.alpha)
        logp = torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = - a * logp * ((1 - p)**self.gamma)
        if self.size_average: return loss.mean()
        else: return loss.sum()

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.65, size_average=True):
        super(FocalBCEWithLogitsLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, logits, targets):
        l = torch.sigmoid(logits).reshape(-1)
        t = targets.reshape(-1)
        p = torch.where(t >= 0.5, l, 1 - l)
        a = torch.where(t >= 0.5, self.alpha, 1 - self.alpha)
        logp = torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = - a * logp * ((1 - p)**self.gamma)
        if self.size_average: return loss.mean()
        else: return loss.sum()