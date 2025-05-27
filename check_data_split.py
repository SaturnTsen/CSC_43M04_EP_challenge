import torch
import pandas as pd
from torch.utils.data import random_split
from data.dataset import Dataset
from data.datamodule import DataModule
import hydra
from configs.experiments.clip_mm import Clip_mmTrainConfig

def check_data_split():
    """检查训练集和验证集划分是否有重叠"""
    
    # 使用相同的配置
    cfg = Clip_mmTrainConfig()
    
    # 创建数据模块
    datamodule = DataModule(
        dataset_path="dataset",  # 使用相对路径
        train_transform=None,  # 不需要变换，只检查ID
        test_transform=None,
        batch_size=cfg.datamodule.batch_size,
        num_workers=0,  # 避免多进程问题
        seed=cfg.datamodule.seed,
        metadata=cfg.datamodule.metadata,
        val_split=cfg.datamodule.val_split,
        standardize_target=cfg.datamodule.standardize_target,
    )
    
    print(f"总数据集大小: {len(datamodule.train_val_dataset)}")
    print(f"训练集大小: {len(datamodule.train_dataset)}")
    print(f"验证集大小: {len(datamodule.val_dataset)}")
    print(f"验证集比例: {cfg.datamodule.val_split}")
    print(f"随机种子: {cfg.datamodule.seed}")
    
    # 获取训练集和验证集的索引
    train_indices = set(datamodule.train_dataset.indices)
    val_indices = set(datamodule.val_dataset.indices)
    
    print(f"\n训练集索引数量: {len(train_indices)}")
    print(f"验证集索引数量: {len(val_indices)}")
    
    # 检查是否有重叠
    overlap = train_indices.intersection(val_indices)
    print(f"重叠的索引数量: {len(overlap)}")
    
    if len(overlap) > 0:
        print("❌ 发现数据泄露！训练集和验证集有重叠")
        print(f"重叠的索引: {list(overlap)[:10]}...")  # 只显示前10个
    else:
        print("✅ 没有数据泄露，训练集和验证集完全分离")
    
    # 检查索引覆盖是否完整
    all_indices = train_indices.union(val_indices)
    expected_indices = set(range(len(datamodule.train_val_dataset)))
    
    if all_indices == expected_indices:
        print("✅ 索引覆盖完整，没有遗漏数据")
    else:
        missing = expected_indices - all_indices
        extra = all_indices - expected_indices
        print(f"❌ 索引覆盖不完整")
        if missing:
            print(f"遗漏的索引: {list(missing)[:10]}...")
        if extra:
            print(f"多余的索引: {list(extra)[:10]}...")
    
    # 获取实际的ID来进一步验证
    print("\n正在检查实际的视频ID...")
    
    # 获取训练集的ID
    train_ids = set()
    for idx in list(train_indices)[:100]:  # 只检查前100个避免太慢
        item = datamodule.train_val_dataset[idx]
        train_ids.add(item['id'])
    
    # 获取验证集的ID
    val_ids = set()
    for idx in list(val_indices)[:100]:  # 只检查前100个避免太慢
    
        item = datamodule.train_val_dataset[idx]
        val_ids.add(item['id'])
    
    # 检查ID重叠
    id_overlap = train_ids.intersection(val_ids)
    print(f"训练集ID样本数: {len(train_ids)}")
    print(f"验证集ID样本数: {len(val_ids)}")
    print(f"ID重叠数量: {len(id_overlap)}")
    
    if len(id_overlap) > 0:
        print("❌ 发现ID重叠！")
        print(f"重叠的ID: {list(id_overlap)[:5]}...")
    else:
        print("✅ 没有ID重叠")
    
    # 检查数据分布
    print("\n检查数据分布...")
    
    # 读取原始CSV文件
    train_val_csv = pd.read_csv("dataset/train_val.csv")
    print(f"原始train_val.csv大小: {len(train_val_csv)}")
    
    # 检查views分布
    views = train_val_csv['views'].values
    print(f"Views统计:")
    print(f"  最小值: {views.min()}")
    print(f"  最大值: {views.max()}")
    print(f"  均值: {views.mean():.2f}")
    print(f"  中位数: {pd.Series(views).median():.2f}")
    
    return train_indices, val_indices, train_ids, val_ids

if __name__ == "__main__":
    check_data_split() 