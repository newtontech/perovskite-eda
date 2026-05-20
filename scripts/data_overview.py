#!/usr/bin/env python3
"""
数据概览脚本
生成 Perovskite Database 的基本统计信息
"""
import pandas as pd
from pathlib import Path
import sys

def main():
    # 数据文件路径
    data_path = Path(__file__).parent.parent / "data" / "raw" / "perovskite_database_all.csv"
    
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先下载数据文件")
        sys.exit(1)
    
    print("📊 Perovskite Database 数据概览")
    print("=" * 60)
    
    # 读取数据
    print("\n⏳ 正在加载数据...")
    df = pd.read_csv(data_path)
    
    # 基本统计
    print(f"\n✅ 数据加载完成！\n")
    print(f"📐 数据维度: {df.shape[0]:,} 行 × {df.shape[1]:,} 列")
    print(f"💾 内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 列信息
    print(f"\n📝 列名和数据类型:")
    print(df.dtypes.to_string())
    
    # 缺失值统计
    print(f"\n❓ 缺失值统计:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        '缺失数量': missing,
        '缺失比例(%)': missing_pct
    })
    print(missing_df[missing_df['缺失数量'] > 0].to_string())
    
    # 数值列统计
    print(f"\n📈 数值列统计:")
    print(df.describe().to_string())
    
    print("\n✅ 概览生成完成！")

if __name__ == "__main__":
    main()
