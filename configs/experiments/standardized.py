from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from configs.experiments.base import BaseTrainConfig
from configs.model.dinov2_improved_standardized import ModelConfig
from configs.optim.adamw import OptimizerConfig
from configs.loss_fn.mse import LossFnConfig
from configs.datamodule.standardized import StandardizedDataModuleConfig

@dataclass
class StandardizedTrainConfig(BaseTrainConfig):
    # 实验名称
    experiment_name: str = "standardized_experiment"
    # 模型配置
    model: ModelConfig = field(default_factory=ModelConfig)
    # 优化器配置
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    # 损失函数配置 (使用标准化MSE)
    loss_fn: LossFnConfig = field(default_factory=LossFnConfig)
    # 数据模块配置 (使用标准化DataModule)
    datamodule: StandardizedDataModuleConfig = field(default_factory=StandardizedDataModuleConfig)
    # 训练轮数
    epochs: int = 30
    # 保存模型的路径
    checkpoint_path: str = "checkpoints/standardized_model.pt"
    # 提交文件路径
    submission_path: str = "submissions/standardized_submission.csv"
    # 是否记录到wandb
    log: bool = True 