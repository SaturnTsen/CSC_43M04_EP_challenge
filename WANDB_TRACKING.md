# WandB 跟踪指标说明

本项目使用WandB进行全面的实验跟踪和可视化。以下是记录的主要指标和信息：

## 1. 训练阶段记录 (train_kfold.py)

### 每个Fold的实验记录
- **实验名称**: `{experiment_name}_fold{fold_idx}`
- **组织**: 所有fold属于同一个group
- **作业类型**: `fold_{fold_idx}`

### 模型信息
- `fold_X/model/total_params`: 模型总参数数量
- `fold_X/model/trainable_params`: 可训练参数数量  
- `fold_X/model/frozen_params`: 冻结参数数量

### 数据信息
- `fold_X/data/train_size`: 训练集大小
- `fold_X/data/val_size`: 验证集大小

### 系统信息
- `fold_X/setup/world_size`: 分布式训练进程数
- `fold_X/setup/device`: 使用的设备

### 训练过程指标
- `fold_X/train/loss_step`: 每步训练损失
- `fold_X/train/loss_epoch`: 每轮训练损失
- `fold_X/val/loss_epoch`: 每轮验证损失
- `fold_X/learning_rate`: 当前学习率

### 评估指标
- `fold_X/val/msle`: 均方对数误差
- `fold_X/val/mae`: 平均绝对误差
- `fold_X/val/mse`: 均方误差
- `fold_X/val/rmse`: 均方根误差

### 预测统计
- `fold_X/predictions/mean`: 预测值均值
- `fold_X/predictions/std`: 预测值标准差
- `fold_X/targets/mean`: 真实值均值
- `fold_X/targets/std`: 真实值标准差

### 模型可解释性
- `fold_X/weights/image`: 图像模态权重 (α)
- `fold_X/weights/text`: 文本模态权重 (β)
- `fold_X/weights/age`: 年龄模态权重 (γ)

### 最佳模型记录
- `fold_X/best_epoch`: 最佳模型对应的轮次
- `fold_X/best_val_loss`: 最佳验证损失
- `fold_X/best_msle`: 最佳MSLE值

### 训练完成信息
- `fold_X/training_completed`: 训练是否完成
- `fold_X/final_best_val_loss`: 最终最佳验证损失
- `fold_X/total_epochs`: 总训练轮数

## 2. Ensemble协调记录

### 主实验记录
- **实验名称**: `{experiment_name}_ensemble`
- **作业类型**: `ensemble_coordination`

### 全局信息
- `ensemble/total_samples`: 总样本数
- `ensemble/n_folds`: fold数量 (3)
- `ensemble/n_gpus`: 可用GPU数量
- `ensemble/world_size`: 使用的进程数

### Fold进度跟踪
- `ensemble/fold_X_start`: fold开始标记
- `ensemble/fold_X_train_size`: fold训练集大小
- `ensemble/fold_X_val_size`: fold验证集大小
- `ensemble/fold_X_completed`: fold完成标记

### 完成状态
- `ensemble/all_folds_completed`: 所有fold是否完成
- `ensemble/ready_for_ensemble`: 是否准备好进行ensemble
- `ensemble/fold_info`: fold信息表格

## 3. 提交阶段记录 (create_submission_ensemble.py)

### 提交实验记录
- **实验名称**: `{experiment_name}_submission`
- **作业类型**: `ensemble_submission`

### Ensemble配置
- `ensemble/n_models`: 参与ensemble的模型数量
- `ensemble/checkpoint_type`: 使用的检查点类型 (best/final)
- `ensemble/method`: ensemble方法 (mean/median/weighted_mean)
- `ensemble/test_samples`: 测试样本数量

### 单模型权重信息
- `ensemble/model_X/image_weight`: 模型X的图像权重
- `ensemble/model_X/text_weight`: 模型X的文本权重
- `ensemble/model_X/age_weight`: 模型X的年龄权重

### Ensemble权重 (如果使用weighted_mean)
- `ensemble/model_X_weight`: 模型X在ensemble中的权重

### 预测统计
- `ensemble/prediction_diversity`: 模型间预测多样性
- `ensemble/pred_std_mean`: 预测标准差均值
- `ensemble/pred_std_max`: 预测标准差最大值
- `ensemble/pred_std_min`: 预测标准差最小值
- `ensemble/final_pred_min`: 最终预测最小值
- `ensemble/final_pred_max`: 最终预测最大值
- `ensemble/final_pred_mean`: 最终预测均值
- `ensemble/final_pred_std`: 最终预测标准差

### 提交信息
- `ensemble/submission_created`: 提交文件是否创建
- `ensemble/submission_path`: 提交文件路径
- `ensemble/model_weights`: 模型权重信息表格

## 4. 可视化建议

### 训练监控
1. **损失曲线**: 比较不同fold的训练和验证损失
2. **权重演化**: 观察各模态权重在训练过程中的变化
3. **MSLE趋势**: 跟踪MSLE指标的改进情况

### 模型分析
1. **权重分布**: 比较不同fold模型的权重分配
2. **预测多样性**: 分析模型间预测的一致性
3. **性能对比**: 比较不同fold的最终性能

### Ensemble效果
1. **预测分布**: 可视化最终预测的分布
2. **模型贡献**: 分析每个模型对ensemble的贡献
3. **不确定性**: 通过预测标准差评估模型置信度

## 5. 使用示例

### 启动训练并记录
```bash
python train_kfold.py --config-name=clip_multimodal
```

### 创建ensemble提交并记录
```bash
python create_submission_ensemble.py --config-name=clip_multimodal
```

### WandB项目设置
- **项目名称**: `challenge_CSC_43M04_EP`
- **实验组织**: 按实验名称分组
- **标签**: 可以通过job_type区分不同阶段

所有实验数据都会自动同步到WandB，便于后续分析和模型选择。 