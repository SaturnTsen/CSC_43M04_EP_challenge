from dataclasses import dataclass, field
from typing import List, Any
from configs.model.dinov2 import ModelConfig
from configs.optim.adamw import OptimizerConfig
from configs.dataset.default import DatasetConfig
from configs.loss_fn.msle import LossFnConfig

@dataclass
class DataModuleConfig:
    _target_: str = "data.datamodule.DataModule"
    dataset_path: str = "${data_dir}"
    train_transform: Any = "${dataset.train_transform}"
    test_transform: Any = "${dataset.val_transform}"
    batch_size: int = "${dataset.batch_size}"
    num_workers: int = "${dataset.num_workers}"
    metadata: List[str] = "${dataset.metadata}"

@dataclass
class TrainConfig:
    epochs: int = 2
    log: bool = True
    prefix: str = ""
    experiment_name: str = "${prefix}${model.name}_${now:%Y-%m-%d_%H-%M-%S}"
    data_dir: str = "${root_dir}/dataset"
    root_dir: str = "${hydra:runtime.cwd}"
    checkpoint_path: str = "${root_dir}/checkpoints/${experiment_name}.pt"
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    loss_fn: LossFnConfig = field(default_factory=LossFnConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)