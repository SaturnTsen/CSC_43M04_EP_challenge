from dataclasses import dataclass, field
from typing import List, Any
from configs.model.dinov2 import ModelConfig
from configs.optim.adamw import OptimizerConfig
from configs.loss_fn.msle import LossFnConfig
from configs.datamodule.default import DataModuleConfig, DatasetConfig


@dataclass
class BaseTrainConfig:
    epochs: int = 20
    log: bool = True
    prefix: str = ""
    experiment_name: str = "${prefix}${model.name}_${now:%Y-%m-%d_%H-%M-%S}"
    data_dir: str = "${root_dir}/dataset"
    root_dir: str = "${hydra:runtime.cwd}"
    checkpoint_path: str = "${root_dir}/checkpoints/${model.name}.pt"
    submission_path: str = "${root_dir}/submissions/${model.name}.csv"
    # MSLE验证频率（每多少个epoch进行一次MSLE验证，设为0表示不验证）
    msle_validation_interval: int = 5
    # MSLE验证结果保存路径 - 使用Hydra的输出目录
    msle_validation_dir: str = "${hydra:run.dir}/validations"
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_fn: LossFnConfig = field(default_factory=LossFnConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig) 
    dataset: DatasetConfig = field(default_factory=DatasetConfig) # only used for explore.ipynb