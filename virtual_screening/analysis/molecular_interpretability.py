#!/usr/bin/env python3
"""
分子可解释性分析
分析高性能分子的共同特征
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

def analyze_top_molecules(
    screened_path: str = 'screened_molecules.csv',
    output_dir: str = 'reports/virtual_screening/figures'
):
    """
    分析 Top 分子的特征
    
    Args:
        screened_path: 筛选结果文件路径
        output_dir: 输出目录
    """
    print(f"🔬 分子可解释性分析")
    print("=" * 60)
    
    # 读取数据
    df = pd.read_csv(screened_path)
    print(f"📖 读取 {len(df)} 个筛选后的分子")
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 骨架分析
    print(f"\n📊 骨架分布分析...")
    scaffold_counts = df['scaffold'].value_counts()
    
    plt.figure(figsize=(10, 6))
    scaffold_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Top 分子骨架分布', fontsize=14, fontweight='bold')
    plt.xlabel('骨架类型', fontsize=12)
    plt.ylabel('分子数量', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaffold_distribution.png', dpi=300)
    print(f"  ✅ 保存: {output_dir}/scaffold_distribution.png")
    
    # 2. 性能分布
    print(f"\n📈 性能分布分析...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PCE 分布
    axes[0, 0].hist(df['predicted_pce'], bins=20, color='green', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('PCE 分布', fontweight='bold')
    axes[0, 0].set_xlabel('PCE (%)')
    axes[0, 0].set_ylabel('分子数量')
    
    # Voc 分布
    axes[0, 1].hist(df['predicted_voc'], bins=20, color='blue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Voc 分布', fontweight='bold')
    axes[0, 1].set_xlabel('Voc (V)')
    axes[0, 1].set_ylabel('分子数量')
    
    # Jsc 分布
    axes[1, 0].hist(df['predicted_jsc'], bins=20, color='orange', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Jsc 分布', fontweight='bold')
    axes[1, 0].set_xlabel('Jsc (mA/cm²)')
    axes[1, 0].set_ylabel('分子数量')
    
    # FF 分布
    axes[1, 1].hist(df['predicted_ff'], bins=20, color='red', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('FF 分布', fontweight='bold')
    axes[1, 1].set_xlabel('FF')
    axes[1, 1].set_ylabel('分子数量')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_distribution.png', dpi=300)
    print(f"  ✅ 保存: {output_dir}/performance_distribution.png")
    
    # 3. 取代基分析
    print(f"\n🔍 取代基分析...")
    sub1_counts = df['substituent_1'].value_counts()
    sub2_counts = df['substituent_2'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sub1_counts.plot(kind='bar', ax=axes[0], color='purple', edgecolor='black')
    axes[0].set_title('取代基 1 分布', fontweight='bold')
    axes[0].set_xlabel('取代基类型')
    axes[0].set_ylabel('分子数量')
    axes[0].tick_params(axis='x', rotation=45)
    
    sub2_counts.plot(kind='bar', ax=axes[1], color='cyan', edgecolor='black')
    axes[1].set_title('取代基 2 分布', fontweight='bold')
    axes[1].set_xlabel('取代基类型')
    axes[1].set_ylabel('分子数量')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/substituent_distribution.png', dpi=300)
    print(f"  ✅ 保存: {output_dir}/substituent_distribution.png")
    
    # 4. 统计报告
    print(f"\n📝 生成统计报告...")
    report = f"""
# SAMs 高通量虚拟筛选报告

## 📊 筛选结果统计

- **总筛选分子数**: {len(df)}
- **平均预测 PCE**: {df['predicted_pce'].mean():.2f}%
- **最高预测 PCE**: {df['predicted_pce'].max():.2f}%
- **平均 SA Score**: {df['sa_score'].mean():.2f}

## 🧪 骨架分布

{scaffold_counts.to_markdown()}

## 🎯 高性能分子特征

### 最常见骨架
- {scaffold_counts.index[0]}: {scaffold_counts.iloc[0]} 个分子 ({scaffold_counts.iloc[0]/len(df)*100:.1f}%)

### 最有效取代基
- 取代基 1: {sub1_counts.index[0]}
- 取代基 2: {sub2_counts.index[0]}

## 📈 性能范围

- **PCE**: {df['predicted_pce'].min():.2f}% - {df['predicted_pce'].max():.2f}%
- **Voc**: {df['predicted_voc'].min():.2f} V - {df['predicted_voc'].max():.2f} V
- **Jsc**: {df['predicted_jsc'].min():.2f} mA/cm² - {df['predicted_jsc'].max():.2f} mA/cm²
- **FF**: {df['predicted_ff'].min():.2f} - {df['predicted_ff'].max():.2f}

## 🏆 Top 10 候选分子

{df[['rank', 'smiles', 'scaffold', 'predicted_pce', 'sa_score']].head(10).to_markdown()}

---

**生成日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    report_path = f'{output_dir}/../screening_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  ✅ 保存: {report_path}")
    print(f"\n✅ 分析完成！共生成 4 张图表和 1 份报告")

if __name__ == "__main__":
    analyze_top_molecules()
