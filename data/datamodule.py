import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional, List, Any, TypedDict

from data.dataset import Dataset
from utils.transforms import TargetStandardizer

class BatchDict(TypedDict):
    id: List[str]
    image: torch.Tensor
    text: List[str]
    target: torch.Tensor
    age_year: torch.Tensor
    
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
        standardize_target: bool = False,
        target_mu: float = 10.50,
        target_sigma: float = 2.20,
        include_age_year: bool = False,
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.val_split = val_split
        self.seed = seed
        self.standardize_target = standardize_target
        self.include_age_year = include_age_year
        
        # 如果启用目标标准化，创建标准化转换器
        self.target_standardizer = None
        if standardize_target:
            self.target_standardizer = TargetStandardizer(mu=target_mu, sigma=target_sigma)
        
        # 初始化数据集
        self.train_val_dataset = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            metadata=self.metadata,
            include_age_year=self.include_age_year,
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

    def standardize_batch(self, batch):
        """应用目标变量标准化"""
        if self.target_standardizer and self.standardize_target:
            batch = self.target_standardizer.standardize(batch)
        return batch

    def collate_fn(self, batch):
        """自定义批处理函数，用于应用目标标准化"""
        # 将列表的字典转换为字典的列表
        result = {}
        for key in batch[0].keys():
            if key == "image":
                result[key] = torch.stack([item[key] for item in batch])
            elif key in ["target", "age_year"]:
                result[key] = torch.tensor([item[key] for item in batch])
            else:
                result[key] = [item[key] for item in batch]
        
        # 应用标准化
        result = self.standardize_batch(result)
        return result

    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
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
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        dataset = Dataset(
            self.dataset_path,
            "test",
            transforms=self.test_transform,
            metadata=self.metadata,
            include_age_year=self.include_age_year,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        
    @property
    def target_standardizer(self):
        return self._target_standardizer
        
    @target_standardizer.setter
    def target_standardizer(self, standardizer):
        self._target_standardizer = standardizer