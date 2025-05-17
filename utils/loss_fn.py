import torch
from torch import nn


class RMSLELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSLELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Ensure the predictions and targets are non-negative
        # -- added a clamp to avoid log(0) and ReLU in the
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        # Compute the RMSLE
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        loss = torch.sqrt(torch.mean((log_pred - log_true) ** 2))

        return loss


class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensure the predictions and targets are non-negative
        # -- added a clamp to avoid log(0) and ReLU in the
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        # Compute the RMSLE
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        loss = torch.mean((log_pred - log_true) ** 2)

        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)

class StandardizedMSELoss(nn.Module):
    """
    标准化的MSE损失
    模型输出标准化后的预测值，而不是直接预测原始值
    """
    def __init__(self):
        super(StandardizedMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.mse(y_pred, y_true)
