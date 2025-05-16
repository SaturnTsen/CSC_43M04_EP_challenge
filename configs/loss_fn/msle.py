from dataclasses import dataclass

@dataclass
class LossFnConfig:
    _target_: str = "utils.loss_fn.MSLELoss"

# 导出配置
LossFnConfig = LossFnConfig