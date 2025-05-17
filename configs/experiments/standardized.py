from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from configs.experiments.base import BaseTrainConfig
from configs.model.dinov2_improved import ModelConfig
from configs.optim.adamw import OptimizerConfig
from configs.loss_fn.msle import LossFnConfig
from configs.datamodule.default import DataModuleConfig

# 修改DataModuleConfig以启用标准化
@dataclass
class StandardizedDataModuleConfig(DataModuleConfig):
    # 启用目标变量标准化
    standardize_target: bool = True
    # log1p(views)的均值和标准差
    target_mu: float = 10.50
    target_sigma: float = 2.20

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