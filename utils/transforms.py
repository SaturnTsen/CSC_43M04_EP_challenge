import torch
import numpy as np
from typing import Dict, Any, Optional

class TargetStandardizer:
    """
    目标变量标准化转换器
    - 训练时: 将原始目标值转换为标准化值 z = (log1p(y) - mu) / sigma
    - 预测时: 将标准化预测值转换回原始值 y_hat = expm1(mu + pred * sigma)
    """
    def __init__(self, mu: float = 10.50, sigma: float = 2.20):
        """
        初始化标准化参数
        Args:
            mu: log1p(views)的均值
            sigma: log1p(views)的标准差
        """
        self.mu = mu
        self.sigma = sigma
        
    def standardize(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        将batch中的target转换为标准化值
        """
        if "target" in batch:
            y = batch["target"]
            if isinstance(y, torch.Tensor):
                # 将原始目标转换为标准化值: z = (log1p(y) - mu) / sigma
                batch["target"] = (torch.log1p(y) - self.mu) / self.sigma
        return batch
    
    def unstandardize(self, pred: torch.Tensor) -> torch.Tensor:
        """
        将标准化预测值转换回原始值
        Args:
            pred: 模型输出的标准化预测值
        Returns:
            原始尺度的预测值
        """
        # 将标准化预测值转换回原始值: y_hat = expm1(mu + pred * sigma)
        return torch.expm1(self.mu + pred * self.sigma) 