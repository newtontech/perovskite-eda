#!/usr/bin/env python3
"""
多目标筛选
基于多个目标（PCE、合成难度、成本）进行分子筛选
"""
import pandas as pd
import numpy as np
from typing import List, Tuple

def multi_objective_screening(
    prediction_path: str = 'prediction_results.csv',
    output_path: str = 'screened_molecules.csv',
    min_pce: float = 20.0,
    max_sa_score: float = 5.0,
    top_n: int = 100
) -> pd.DataFrame:
    """
    多目标筛选
    
    Args:
        prediction_path: 预测结果文件路径
        output_path: 输出文件路径
        min_pce: 最小 PCE 阈值
        max_sa_score: 最大 SA Score 阈值
        top_n: 保留前 N 个分子
    
    Returns:
        筛选后的 DataFrame
    """
    print(f"🎯 多目标筛选")
    print("=" * 60)
    
    # 读取预测结果
    print(f"📖 读取预测结果: {prediction_path}")
    df = pd.read_csv(prediction_path)
    print(f"  - 总分子数: {len(df)}")
    
    # 筛选条件
    print(f"\n🔍 应用筛选条件:")
    print(f"  - PCE > {min_pce}%")
    print(f"  - SA Score < {max_sa_score}")
    
    # 应用筛选
    filtered = df[
        (df['predicted_pce'] >= min_pce) &
        (df['sa_score'] <= max_sa_score)
    ].copy()
    
    print(f"\n✅ 筛选后分子数: {len(filtered)}")
    
    if len(filtered) == 0:
        print("⚠️  没有分子满足筛选条件")
        return pd.DataFrame()
    
    # 多目标排序
    print(f"\n📊 多目标排序...")
    
    # 归一化各个目标
    filtered['pce_score'] = (filtered['predicted_pce'] - filtered['predicted_pce'].min()) / \
                            (filtered['predicted_pce'].max() - filtered['predicted_pce'].min())
    
    filtered['sa_score_norm'] = 1 - (filtered['sa_score'] - filtered['sa_score'].min()) / \
                                (filtered['sa_score'].max() - filtered['sa_score'].min())
    
    # 综合评分（PCE 权重 0.7，合成难度权重 0.3）
    filtered['composite_score'] = 0.7 * filtered['pce_score'] + 0.3 * filtered['sa_score_norm']
    
    # 排序
    filtered = filtered.sort_values('composite_score', ascending=False)
    
    # 选取 Top N
    top_molecules = filtered.head(top_n).copy()
    
    # 添加排名
    top_molecules['rank'] = range(1, len(top_molecules) + 1)
    
    # 保存结果
    top_molecules.to_csv(output_path, index=False)
    
    # 统计
    print(f"\n🏆 Top {len(top_molecules)} 分子:")
    print(f"  - 平均 PCE: {top_molecules['predicted_pce'].mean():.2f}%")
    print(f"  - 平均 SA Score: {top_molecules['sa_score'].mean():.2f}")
    print(f"  - 平均综合评分: {top_molecules['composite_score'].mean():.3f}")
    print(f"💾 结果保存到: {output_path}")
    
    # 显示 Top 10
    print(f"\n🥇 Top 10 分子:")
    print(top_molecules[['rank', 'smiles', 'predicted_pce', 'sa_score', 'composite_score']].head(10))
    
    return top_molecules

if __name__ == "__main__":
    multi_objective_screening()
