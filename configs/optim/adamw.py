from dataclasses import dataclass, field
from typing import List

@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.AdamW"
    lr: float = 1e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.01

# 导出配置
OptimizerConfig = OptimizerConfig