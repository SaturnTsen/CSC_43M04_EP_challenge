from dataclasses import dataclass, field

from configs.experiments.base import BaseTrainConfig
from configs.model.clip_multimodal import ModelConfig
from configs.optim.adamw import OptimizerConfig
from configs.loss_fn.msle import LossFnConfig
from configs.datamodule.default import DataModuleConfig


@dataclass
class Clip_mmTrainConfig(BaseTrainConfig):
    """使用 CLIP 多模态模型的训练配置 (不使用目标标准化)"""

    # 实验名称
    experiment_name: str = "clip_mm_experiment"

    # 模型配置
    model: ModelConfig = field(default_factory=ModelConfig)

    # 优化器
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    # 损失函数 (MSLE 直接在原始尺度上评估)
    loss_fn: LossFnConfig = field(default_factory=LossFnConfig)

    # 数据模块 (保持默认，不启用标准化)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)

    # 训练轮数
    epochs: int = 20

    # 检查点 & 提交路径
    checkpoint_path: str = "checkpoints/clip_mm.pt"
    submission_path: str = "submissions/clip_mm.csv"

    # 记录到 wandb
    log: bool = True

    # MSLE 验证间隔
    msle_validation_interval: int = 3
    msle_validation_dir: str = "${hydra:run.dir}/validations" 