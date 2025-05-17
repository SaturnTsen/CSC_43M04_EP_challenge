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

class WeightedStandardizedMSELoss(nn.Module):
    """
    标准化回归下的加权 MSE，用于增强爆款（高 log1p(y)）样本惩罚。
    使用双阈值 sigmoid 平滑加权策略：13.5 启动，16.5 饱和。
    """
    def __init__(self, mean, std, lambda_weight=2.0, lower=13.5, upper=16.5):
        super(WeightedStandardizedMSELoss, self).__init__()
        self.mean = mean
        self.std = std
        self.lambda_weight = lambda_weight
        self.lower = lower
        self.upper = upper
        self.mse = nn.MSELoss(reduction='none')  # 逐点计算

    def forward(self, y_pred, y_true_std):
        # 反标准化：z-score → log1p(y)
        y_true_log = y_true_std * self.std + self.mean

        # 权重函数：双阈值 sigmoid，center = (lower + upper)/2
        center = 0.5 * (self.lower + self.upper)
        sharpness = 6.0 / (self.upper - self.lower)  # sigmoid在 [lower, upper] 区间内基本完成上升
        weight = 1.0 + self.lambda_weight * torch.sigmoid((y_true_log - center) * sharpness)

        # MSE loss in standardized space
        loss = self.mse(y_pred, y_true_std)

        # 加权平均
        return (weight * loss).mean()
