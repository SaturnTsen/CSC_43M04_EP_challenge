import torch
import pandas as pd
from PIL import Image
from datetime import datetime


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, metadata, transforms=None):
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        print(f"Reading {dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")
        
        # 计算年份差值
        current_year = datetime.now().year
        info["year"] = pd.to_datetime(info["date"]).dt.year
        info["age_years"] = current_year - info["year"]
        
        # 分别存储title和description
        self.titles = info["title"].values
        self.descriptions = info["description"].values
        self.age_years = torch.tensor(info["age_years"].values, dtype=torch.float32)
        
        if "views" in info.columns:
            self.targets = info["views"].values

        # - ids
        self.ids = info["id"].values
        
        # - transforms
        self.transforms = transforms

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        # - load the image
        image = Image.open(
            f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg"
        ).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
            
        value = {
            "id": self.ids[idx],
            "image": image,
            "title": self.titles[idx],
            "description": self.descriptions[idx],
            "age_years": self.age_years[idx],
        }
        
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value
