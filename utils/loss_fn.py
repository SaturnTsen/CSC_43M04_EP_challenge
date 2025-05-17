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
    支持lambda scheduler：lambda_now = min(lambda_max, lambda_max * step / warmup_steps)
    """
    def __init__(self, mean, std, lambda_weight, lower, upper, lambda_max=None, warmup_steps=None):
        super(WeightedStandardizedMSELoss, self).__init__()
        self.mean = mean
        self.std = std
        self.lambda_weight = lambda_weight  # 初始lambda值或固定值（如果不使用scheduler）
        self.lower = lower
        self.upper = upper
        self.mse = nn.MSELoss(reduction='none')  # 逐点计算
        
        # Lambda scheduler 参数
        self.lambda_max = lambda_max if lambda_max is not None else lambda_weight
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.current_lambda = lambda_weight  # 当前使用的lambda值

    def step(self):
        """
        每次训练步骤后更新lambda值
        """
        if self.warmup_steps is not None and self.warmup_steps > 0:
            self.current_step += 1
            self.current_lambda = min(
                self.lambda_max, 
                self.lambda_max * self.current_step / self.warmup_steps
            )
            return self.current_lambda
        return self.lambda_weight
    
    def get_current_lambda(self):
        """
        获取当前lambda值
        """
        if self.warmup_steps is not None and self.warmup_steps > 0:
            return self.current_lambda
        return self.lambda_weight

    def forward(self, y_pred, y_true_std):
        # 反标准化：z-score → log1p(y)
        y_true_log = y_true_std * self.std + self.mean

        # 使用当前lambda值
        current_lambda = self.get_current_lambda()
        
        # 权重函数：双阈值 sigmoid，center = (lower + upper)/2
        center = 0.5 * (self.lower + self.upper)
        sharpness = 6.0 / (self.upper - self.lower)  # sigmoid在 [lower, upper] 区间内基本完成上升
        weight = 1.0 + current_lambda * torch.sigmoid((y_true_log - center) * sharpness)

        # MSE loss in standardized space
        loss = self.mse(y_pred, y_true_std)

        # 加权平均
        return (weight * loss).mean()
