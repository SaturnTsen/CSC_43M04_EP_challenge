import torch
import pandas as pd
from torch.utils.data import random_split
from data.dataset import Dataset

def simple_data_check():
    """简单检查数据集划分"""
    
    print("=== 数据集划分检查 ===")
    
    # 创建数据集（不使用变换）
    dataset = Dataset(
        dataset_path="dataset",
        split="train_val",
        metadata=["title"],
        transforms=None
    )
    
    print(f"总数据集大小: {len(dataset)}")
    
    # 使用相同的参数进行划分
    val_split = 0.1
    seed = 42
    
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    print(f"验证集比例: {val_split}")
    print(f"随机种子: {seed}")
    
    # 划分数据集
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # 获取索引
    train_indices = set(train_dataset.indices)
    val_indices = set(val_dataset.indices)
    
    print(f"\n实际训练集索引数量: {len(train_indices)}")
    print(f"实际验证集索引数量: {len(val_indices)}")
    
    # 检查重叠
    overlap = train_indices.intersection(val_indices)
    print(f"重叠的索引数量: {len(overlap)}")
    
    if len(overlap) > 0:
        print("❌ 发现数据泄露！训练集和验证集有重叠")
        return False
    else:
        print("✅ 没有数据泄露，训练集和验证集完全分离")
    
    # 检查覆盖
    all_indices = train_indices.union(val_indices)
    expected_indices = set(range(len(dataset)))
    
    if all_indices == expected_indices:
        print("✅ 索引覆盖完整")
    else:
        print("❌ 索引覆盖不完整")
        return False
    
    # 检查一些实际的ID
    print("\n检查实际ID...")
    
    # 获取一些训练集ID
    train_ids = []
    for i, idx in enumerate(list(train_indices)[:10]):
        item = dataset[idx]
        train_ids.append(item['id'])
    
    # 获取一些验证集ID
    val_ids = []
    for i, idx in enumerate(list(val_indices)[:10]):
        item = dataset[idx]
        val_ids.append(item['id'])
    
    print(f"训练集前10个ID: {train_ids}")
    print(f"验证集前10个ID: {val_ids}")
    
    # 检查ID重叠
    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    id_overlap = train_id_set.intersection(val_id_set)
    
    if len(id_overlap) > 0:
        print(f"❌ 发现ID重叠: {id_overlap}")
        return False
    else:
        print("✅ 前10个ID没有重叠")
    
    # 读取原始数据统计
    print("\n=== 原始数据统计 ===")
    train_val_csv = pd.read_csv("dataset/train_val.csv")
    test_csv = pd.read_csv("dataset/test.csv")
    
    print(f"train_val.csv 大小: {len(train_val_csv)}")
    print(f"test.csv 大小: {len(test_csv)}")
    
    # 检查views分布
    views = train_val_csv['views'].values
    print(f"\nViews分布:")
    print(f"  最小值: {views.min():,}")
    print(f"  最大值: {views.max():,}")
    print(f"  均值: {views.mean():,.2f}")
    print(f"  中位数: {pd.Series(views).median():,.2f}")
    print(f"  标准差: {views.std():,.2f}")
    
    # 检查是否有重复ID
    train_val_ids = set(train_val_csv['id'].values)
    test_ids = set(test_csv['id'].values)
    
    print(f"\ntrain_val 唯一ID数量: {len(train_val_ids)}")
    print(f"test 唯一ID数量: {len(test_ids)}")
    
    # 检查train_val和test之间是否有重叠
    train_test_overlap = train_val_ids.intersection(test_ids)
    if len(train_test_overlap) > 0:
        print(f"❌ train_val和test之间有ID重叠: {len(train_test_overlap)}个")
        print(f"重叠ID示例: {list(train_test_overlap)[:5]}")
        return False
    else:
        print("✅ train_val和test之间没有ID重叠")
    
    return True

if __name__ == "__main__":
    success = simple_data_check()
    if success:
        print("\n🎉 数据集划分检查通过！")
    else:
        print("\n⚠️  数据集划分存在问题！") 