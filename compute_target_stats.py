import pandas as pd
import numpy as np

def compute_target_statistics(dataset_path: str = "dataset/train_val.csv"):
    """计算目标变量(views)的标准化参数"""
    
    print(f"Reading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # 获取views列
    views = df['views'].values
    
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(views)}")
    print(f"  Views range: {views.min():.0f} - {views.max():.0f}")
    print(f"  Views mean: {views.mean():.2f}")
    print(f"  Views std: {views.std():.2f}")
    
    # 计算log1p变换后的统计量
    log_views = np.log1p(views)
    
    print(f"\nLog1p(views) statistics:")
    print(f"  Range: {log_views.min():.4f} - {log_views.max():.4f}")
    print(f"  Mean (mu): {log_views.mean():.4f}")
    print(f"  Std (sigma): {log_views.std():.4f}")
    
    # 返回标准化参数
    return log_views.mean(), log_views.std()

if __name__ == "__main__":
    mu, sigma = compute_target_statistics()
    
    print(f"\n=== 标准化参数 ===")
    print(f"target_mu: {mu:.4f}")
    print(f"target_sigma: {sigma:.4f}")
    
    print(f"\n=== 配置文件中使用 ===")
    print(f"target_mu={mu:.4f}")
    print(f"target_sigma={sigma:.4f}") 