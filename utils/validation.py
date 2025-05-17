import torch
import pandas as pd
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from tqdm import tqdm
from datetime import datetime
import csv

from utils.transforms import TargetStandardizer
from utils.loss_fn import MSLELoss
from data.datamodule import BatchDict

def validate_and_log(
    model: nn.Module, 
    val_loader: DataLoader, 
    device: torch.device, 
    epoch: int, 
    target_standardizer: Optional[TargetStandardizer] = None, 
    experiment_name: str = "experiment",
    save_dir: str = "outputs/validations",
    log_wandb: bool = True,
    logger = None
) -> Tuple[float, pd.DataFrame]:
    """
    验证模型性能，计算MSLE，并记录预测结果
    
    Args:
        model: 训练的模型
        val_loader: 验证数据加载器
        device: 使用的设备
        epoch: 当前的epoch
        target_standardizer: 标准化工具（如果使用了标准化）
        experiment_name: 实验名称
        save_dir: 结果保存目录
        log_wandb: 是否记录到wandb
        logger: wandb logger实例
        
    Returns:
        msle: MSLE损失值
        results_df: 包含预测结果的DataFrame
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化MSLE损失函数
    msle_fn = MSLELoss()
    
    # 为结果创建记录
    records = []
    
    # 设置模型为评估模式
    model.eval()
    
    # 存储所有预测值和真实值用于计算MSLE
    all_preds = []
    all_targets = []
    
    # 遍历验证集
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch}", leave=False):
            batch: BatchDict = batch
            batch["image"] = batch["image"].to(device)
            
            # 获取原始目标值（用于计算MSLE）
            original_targets = batch["target"].clone()
            
            # 目标值移动到设备
            batch["target"] = batch["target"].to(device).squeeze()
            
            # 获取模型预测
            preds = model(batch).squeeze()
            
            # 如果使用标准化，将预测值转换回原始尺度
            unstandardized_preds = preds
            if target_standardizer:
                unstandardized_preds = target_standardizer.unstandardize(preds)
            
            # 转到CPU
            unstandardized_preds = unstandardized_preds.cpu()
            
            # 收集所有预测值和目标值用于计算整体MSLE
            all_preds.append(unstandardized_preds)
            all_targets.append(original_targets.squeeze())
            
            # 创建记录
            for idx, id_, pred, target in zip(
                range(len(unstandardized_preds)), 
                batch["id"], 
                unstandardized_preds.numpy(), 
                original_targets.squeeze().numpy()
            ):
                records.append({
                    "ID": id_.item() if isinstance(id_, torch.Tensor) else id_,
                    "epoch": epoch,
                    "predicted_views": pred.item() if isinstance(pred, np.ndarray) else pred,
                    "actual_views": target.item() if isinstance(target, np.ndarray) else target
                })
    
    # 转换为tensor用于计算MSLE
    all_preds_tensor = torch.cat(all_preds)
    all_targets_tensor = torch.cat(all_targets)
    
    # 计算MSLE
    msle = msle_fn(all_preds_tensor, all_targets_tensor).item()
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(records)
    
    # 保存结果到CSV
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = os.path.join(save_dir, f"{experiment_name}_epoch{epoch}_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    
    # 使用CSV记录每个epoch的MSLE
    msle_log_path = os.path.join(save_dir, f"{experiment_name}_msle_log.csv")
    
    # 如果文件不存在，创建并添加表头
    if not os.path.exists(msle_log_path):
        with open(msle_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['experiment_name', 'epoch', 'msle', 'timestamp'])
    
    # 添加新的记录
    with open(msle_log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([experiment_name, epoch, msle, timestamp])
    
    # 记录CSV路径到wandb (但不记录MSLE值，避免与train.py重复)
    if log_wandb and logger is not None:
        logger.log({
            "val/msle_csv_path": results_path
        })
    
    print(f"Epoch {epoch} Validation MSLE: {msle:.6f}")
    print(f"Saved validation results to {results_path}")
    print(f"Updated MSLE log at {msle_log_path}")
    
    return msle, results_df 