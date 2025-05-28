from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class TestTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"_target_": "torchvision.transforms.Resize", "size": [336, 336]},  # CLIP Large的输入尺寸
        {"_target_": "torchvision.transforms.ToTensor"},
        {"_target_": "torchvision.transforms.Normalize",
         "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]}  # CLIP的标准化参数
    ]) 