from dataclasses import dataclass

@dataclass
class LossFnConfig:
    _target_: str = "utils.loss_fn.WeightedStandardizedMSELoss"
    mean: float = 10.50
    std: float = 2.20
    lambda_weight: float = 2.0
    lower: float = 13.5
    upper: float = 16.5
    lambda_max: float = 1.3  # lambda的最大值
    warmup_steps: int = 1300  # 预热步数，设置为None或0可以禁用scheduler

# 导出配置
LossFnConfig = LossFnConfig 