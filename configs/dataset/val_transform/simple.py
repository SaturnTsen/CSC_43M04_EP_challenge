from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ValTransformConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"_target_": "torchvision.transforms.Resize", "size": [224, 224]},
        {"_target_": "torchvision.transforms.ToTensor"},
        {"_target_": "torchvision.transforms.Normalize",
         "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    ])