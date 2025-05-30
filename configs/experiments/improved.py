from dataclasses import dataclass, field
from typing import List, Any
from configs.model.dinov2_improved import ModelConfig
from configs.optim.adamw import OptimizerConfig
from configs.loss_fn.msle import LossFnConfig
from configs.datamodule.default import DataModuleConfig
from configs.experiments.base import BaseTrainConfig

@dataclass
class ImprovedTrainConfig(BaseTrainConfig):
    epochs: int = 20
    prefix: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_fn: LossFnConfig = field(default_factory=LossFnConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)