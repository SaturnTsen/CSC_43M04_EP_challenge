from dataclasses import dataclass, field
from typing import List, Any
from configs.model.clip_multimodal import ModelConfig
from configs.optim.adamw import OptimizerConfig
from configs.loss_fn.msle import LossFnConfig
from configs.datamodule.clip import DataModuleConfig, DatasetConfig
from configs.experiments.base import BaseTrainConfig

@dataclass
class CLIPMultiModalTrainConfig(BaseTrainConfig):
    epochs: int = 30
    prefix: str = "clip_"
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_fn: LossFnConfig = field(default_factory=LossFnConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig) 