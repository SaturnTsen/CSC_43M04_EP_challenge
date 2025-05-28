import torch
import torch.nn as nn
import clip
from typing import Dict, Any
import torch.nn.functional as F


class CLIPMultimodalModel(nn.Module):
    """
    多模态CLIP模型，结合图像、文本描述和年龄信息进行视频观看量预测
    """
    def __init__(
        self, 
        clip_model: str = "ViT-L/14",  # 使用大型CLIP模型
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_clip: bool = False
    ):
        super().__init__()
        
        # 加载CLIP模型
        self.clip_model, _ = clip.load(clip_model, device="cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取CLIP的特征维度
        self.clip_dim = self.clip_model.visual.output_dim  # 通常是768对于ViT-L/14
        
        # 是否冻结CLIP参数
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # 三个独立的回归头
        self.image_head = nn.Sequential(
            nn.Linear(self.clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.text_head = nn.Sequential(
            nn.Linear(self.clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 年龄特征的嵌入层和回归头
        self.age_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        self.age_head = nn.Sequential(
            nn.Linear(128, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 可学习的权重参数
        self.alpha = nn.Parameter(torch.tensor(0.4))  # 图像权重
        self.beta = nn.Parameter(torch.tensor(0.4))   # 文本权重
        # gamma = 1 - alpha - beta (年龄权重)
        
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # 提取输入
        images = batch["image"]
        texts = batch["text"]
        ages = batch["age_year"].float().unsqueeze(-1)  # [B, 1]
        
        # 使用CLIP编码图像和文本
        with torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            text_tokens = clip.tokenize(texts, truncate=True).to(images.device)
            text_features = self.clip_model.encode_text(text_tokens)
        
        # 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 通过各自的回归头
        image_pred = self.image_head(image_features)
        text_pred = self.text_head(text_features)
        
        # 年龄特征处理
        age_features = self.age_embedding(ages)
        age_pred = self.age_head(age_features)
        
        # 使用softmax确保权重和为1
        weights = F.softmax(torch.stack([self.alpha, self.beta, 
                                         torch.tensor(1.0).to(self.alpha.device) - self.alpha - self.beta]), dim=0)
        
        # 加权组合预测
        final_pred = weights[0] * image_pred + weights[1] * text_pred + weights[2] * age_pred
        
        # 保存各个组件的预测和权重用于分析
        self.last_predictions = {
            'image': image_pred,
            'text': text_pred,
            'age': age_pred,
            'weights': weights,
            'final': final_pred
        }
        
        return final_pred
    
    def get_interpretability_info(self):
        """获取模型可解释性信息"""
        if hasattr(self, 'last_predictions'):
            weights = F.softmax(torch.stack([self.alpha, self.beta, 
                                            torch.tensor(1.0).to(self.alpha.device) - self.alpha - self.beta]), dim=0)
            return {
                'image_weight': weights[0].item(),
                'text_weight': weights[1].item(),
                'age_weight': weights[2].item()
            }
        return None 