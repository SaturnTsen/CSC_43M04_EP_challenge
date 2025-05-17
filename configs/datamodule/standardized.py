from dataclasses import dataclass
from configs.datamodule.default import DataModuleConfig

# 修改DataModuleConfig以启用标准化
@dataclass
class StandardizedDataModuleConfig(DataModuleConfig):
    # 启用目标变量标准化
    standardize_target: bool = True
    # log1p(views)的均值和标准差
    target_mu: float = 10.50
    target_sigma: float = 2.20