#!/usr/bin/env python3
"""
测试脚本：验证包含 age_year 特征的 CLIP 模型是否工作正常
"""

import torch
import numpy as np
from models.clip_multimodal_with_age import CLIPMultiModalWithAgeRegressor
from models.clip_multimodal_with_age_standardized import CLIPMultiModalWithAgeRegressorStandardized

def test_age_model():
    """测试包含 age_year 特征的模型"""
    
    device = torch.device("cpu")  # 使用CPU进行测试
    
    # 测试非标准化版本
    print("测试非标准化版本模型...")
    model = CLIPMultiModalWithAgeRegressor(
        clip_model_name="ViT-B-32",
        hidden_dim=256,
        dropout=0.1
    ).to(device)
    
    # 创建测试数据
    batch_size = 4
    test_batch = {
        "image": torch.randn(batch_size, 3, 224, 224).to(device),
        "text": ["Test video title 1", "Test video title 2", "Test video title 3", "Test video title 4"],
        "age_year": torch.tensor([1.0, 2.5, 0.5, 4.0]).to(device)  # 年份特征
    }
    
    # 前向传播测试
    model.eval()
    with torch.no_grad():
        predictions = model(test_batch)
        print(f"预测输出形状: {predictions.shape}")
        print(f"预测值: {predictions.squeeze().tolist()}")
        
        # 测试融合权重
        fusion_weights = model.get_fusion_weights()
        print(f"融合权重: {fusion_weights}")
        
        # 测试单独预测
        img_pred, text_pred, age_pred = model.get_individual_predictions(test_batch)
        print(f"图像预测: {img_pred.squeeze().tolist()}")
        print(f"文本预测: {text_pred.squeeze().tolist()}")
        print(f"年份预测: {age_pred.squeeze().tolist()}")
    
    print("\n" + "="*50)
    
    # 测试标准化版本
    print("测试标准化版本模型...")
    model_std = CLIPMultiModalWithAgeRegressorStandardized(
        clip_model_name="ViT-B-32",
        hidden_dim=256,
        dropout=0.1
    ).to(device)
    
    model_std.eval()
    with torch.no_grad():
        predictions_std = model_std(test_batch)
        print(f"标准化模型预测输出形状: {predictions_std.shape}")
        print(f"标准化模型预测值: {predictions_std.squeeze().tolist()}")
        
        fusion_weights_std = model_std.get_fusion_weights()
        print(f"标准化模型融合权重: {fusion_weights_std}")
    
    print("\n测试完成！所有模型都能正常工作。")

if __name__ == "__main__":
    test_age_model() 