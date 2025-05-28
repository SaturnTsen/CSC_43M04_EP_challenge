from dataclasses import dataclass, field

from configs.experiments.base import BaseTrainConfig
from configs.model.clip_multimodal_standardized import ModelConfig
from configs.optim.adamw import OptimizerConfig
from configs.loss_fn.mse import LossFnConfig
from configs.datamodule.standardized import StandardizedDataModuleConfig


@dataclass
class Clip_mm_standardizedTrainConfig(BaseTrainConfig):
    """使用 CLIP 多模态模型 + 目标标准化的训练配置"""

    # 实验名称
    experiment_name: str = "clip_mm_standardized_experiment"

    # 模型配置 (使用 CLIP 多模态)
    model: ModelConfig = field(default_factory=ModelConfig)

    # 优化器配置
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    # 损失函数配置 (使用标准化MSE，因为目标已经标准化)
    loss_fn: LossFnConfig = field(default_factory=LossFnConfig)

    # 数据模块配置 (使用标准化DataModule)
    datamodule: StandardizedDataModuleConfig = field(default_factory=StandardizedDataModuleConfig)

    # 训练轮数
    epochs: int = 40

    # 检查点 & 提交路径
    checkpoint_path: str = "checkpoints/clip_mm_standardized.pt"
    submission_path: str = "submissions/clip_mm_standardized.csv"

    # 记录到 wandb
    log: bool = True

    # MSLE 验证间隔
    msle_validation_interval: int = 3
    msle_validation_dir: str = "${hydra:run.dir}/validations" 