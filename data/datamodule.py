import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional, List, Any, TypedDict

from data.dataset import Dataset

class BatchDict(TypedDict):
    id: List[str]
    image: torch.Tensor
    text: List[str]
    target: torch.Tensor
    
class DataModule:
    def __init__(
        self,
        dataset_path: str,
        train_transform: Any,
        test_transform: Any,
        batch_size: int,
        num_workers: int,
        seed: int,
        metadata: List[str] = ["title"],
        val_split: float = 0.1,
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.val_split = val_split
        self.seed = seed
        
        # 初始化数据集
        self.train_val_dataset = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            metadata=self.metadata,
        )
        
        # 计算训练集和验证集的大小
        total_size = len(self.train_val_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size
        
        # 划分数据集
        self.train_dataset, self.val_dataset = random_split(
            self.train_val_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """验证数据加载器"""
        if self.val_dataset is None:
            return None
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        dataset = Dataset(
            self.dataset_path,
            "test",
            transforms=self.test_transform,
            metadata=self.metadata,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )