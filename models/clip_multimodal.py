import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class CLIPMultimodalModel(nn.Module):
    """
    多模态CLIP模型，结合图像、文本描述和年龄信息进行视频观看量预测
    - 使用transformers的CLIP模型
    - 将age作为额外token嵌入到特征中
    """
    def __init__(
        self, 
        clip_model: str = "openai/clip-vit-large-patch14",  # 使用transformers的CLIP模型
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_clip: bool = True
    ):
        super().__init__()
        
        # 加载CLIP模型和处理器
        self.clip = CLIPModel.from_pretrained(clip_model)
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        
        # 获取CLIP的特征维度
        self.clip_dim = self.clip.config.projection_dim  # 通常是768对于ViT-L/14
        
        # 是否冻结CLIP参数
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        # 年龄特征的嵌入层
        self.age_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, self.clip_dim),  # 映射到与CLIP特征相同的维度
            nn.LayerNorm(self.clip_dim)
        )
        
        # 融合层 - 将图像、文本和年龄特征融合
        self.fusion = nn.Sequential(
            nn.Linear(self.clip_dim * 3, hidden_dim),  # 3倍CLIP_dim因为我们concat三个特征
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # 提取输入
        images = batch["image"]  # [B, C, H, W]
        texts = batch["text"]    # [B]
        ages = batch["age_year"].float().unsqueeze(-1)  # [B, 1]
        
        # 使用CLIP处理图像和文本
        with torch.amp.autocast():
            # 获取CLIP特征
            outputs = self.clip(
                input_ids=self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).input_ids.to(images.device),
                pixel_values=images,
                return_dict=True
            )
            
            # 提取图像和文本特征
            image_features = outputs.image_embeds      # [B, clip_dim]
            text_features = outputs.text_embeds        # [B, clip_dim]
        
        # 生成年龄特征
        age_features = self.age_embedding(ages)        # [B, clip_dim]
        
        # 归一化所有特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        age_features = F.normalize(age_features, dim=-1)
        
        # 拼接所有特征
        combined_features = torch.cat([
            image_features,
            text_features,
            age_features
        ], dim=1)  # [B, clip_dim * 3]
        
        # 通过融合层得到最终预测
        predictions = self.fusion(combined_features)
        
        # 保存特征用于分析
        self.last_features = {
            'image': image_features,
            'text': text_features,
            'age': age_features,
            'combined': combined_features,
            'predictions': predictions
        }
        
        return predictions
    
    def get_feature_similarities(self):
        """获取不同模态特征之间的相似度，用于分析"""
        if not hasattr(self, 'last_features'):
            return None
            
        image_text_sim = F.cosine_similarity(
            self.last_features['image'],
            self.last_features['text']
        ).mean().item()
        
        image_age_sim = F.cosine_similarity(
            self.last_features['image'],
            self.last_features['age']
        ).mean().item()
        
        text_age_sim = F.cosine_similarity(
            self.last_features['text'],
            self.last_features['age']
        ).mean().item()
        
        return {
            'image_text_similarity': image_text_sim,
            'image_age_similarity': image_age_sim,
            'text_age_similarity': text_age_sim
        } 