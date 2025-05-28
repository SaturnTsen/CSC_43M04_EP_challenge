# CLIP多模态模型使用指南

本项目实现了一个基于CLIP的多模态模型，用于YouTube视频观看量预测。该模型结合了图像（缩略图）、文本（标题+描述）和时间（视频年龄）三个模态的信息。

## 主要特性

### 1. 多模态输入
- **图像**：使用CLIP的视觉编码器处理视频缩略图
- **文本**：使用CLIP的文本编码器处理标题和描述的组合
- **年龄**：计算当前年份与上传年份的差值

### 2. 可解释性设计
- 三个独立的回归头分别处理不同模态
- 可学习的权重参数（α, β, γ）控制各模态的贡献
- 权重满足约束：α + β + γ = 1

### 3. 3-fold交叉验证
- 自动将数据集分成3折
- 支持多GPU并行训练（每个GPU训练一个fold）
- 自动保存最佳和最终模型

### 4. 模型集成
- 支持多个fold模型的ensemble
- 提供mean、median和weighted_mean三种集成方法

## 安装依赖

```bash
pip install -r requirements.txt
```

注意：CLIP需要额外安装：
```bash
pip install git+https://github.com/openai/CLIP.git
```

## 使用方法

### 1. 3-fold交叉验证训练

```bash
python train_kfold.py --config-name=clip_multimodal
```

这将：
- 自动创建3个fold
- 在可用的GPU上并行训练（最多使用3个GPU）
- 保存每个fold的最佳模型和最终模型
- 使用WandB记录训练过程

### 2. 创建ensemble提交

```bash
python create_submission_ensemble.py --config-name=clip_multimodal
```

可选参数：
- `use_best_checkpoint=true/false`：使用最佳模型还是最终模型
- `ensemble_method=mean/median/weighted_mean`：集成方法
- `ensemble_weights=[0.4,0.3,0.3]`：如果使用weighted_mean

### 3. 查看模型权重

训练过程中，模型会自动记录三个模态的权重到WandB：
- `fold_X/weights/image`：图像模态权重
- `fold_X/weights/text`：文本模态权重  
- `fold_X/weights/age`：年龄模态权重

## 配置说明

主要配置文件：`configs/experiments/clip_multimodal.py`

关键参数：
- `model.instance.clip_model`：CLIP模型版本（默认"ViT-L/14"）
- `model.instance.freeze_clip`：是否冻结CLIP参数
- `datamodule.batch_size`：批次大小（根据显存调整）
- `epochs`：训练轮数
- `optim.lr`：学习率

## 模型架构

```
输入 → CLIP编码器 → 模态特定回归头 → 加权组合 → 最终预测
  ├─ 图像 → CLIP视觉编码器 → 图像回归头 ─┐
  ├─ 文本 → CLIP文本编码器 → 文本回归头 ─┼─ α·pred_img + β·pred_text + γ·pred_age
  └─ 年龄 → 嵌入层 → 年龄回归头 ────────┘
```

## 性能优化建议

1. **显存管理**：
   - RTX 3090 (24GB)建议batch_size=16
   - 如果显存不足，可以设置`freeze_clip=true`冻结CLIP参数

2. **训练速度**：
   - 使用多GPU并行训练3个fold
   - 调整`num_workers`优化数据加载

3. **模型性能**：
   - 尝试不同的CLIP模型版本
   - 调整`hidden_dim`和`dropout`
   - 使用不同的ensemble方法

## 输出文件

训练完成后会生成：
- `checkpoints/clip_multimodal_fold{0,1,2}_best.pt`：各fold最佳模型
- `checkpoints/clip_multimodal_fold{0,1,2}_final.pt`：各fold最终模型
- `submissions/clip_multimodal_ensemble_mean.csv`：提交文件
- `fold_info.csv`：fold划分信息

## 注意事项

1. 确保数据集包含`title`、`description`和`year`字段
2. 第一次运行时CLIP模型下载可能需要一些时间
3. 多GPU训练需要确保所有GPU可用且型号相同 