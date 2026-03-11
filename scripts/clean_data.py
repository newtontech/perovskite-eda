#!/usr/bin/env python3
"""
钙钛矿数据库数据清洗脚本
Data Cleaning Script for Perovskite Database

作者: OpenClaw AI Assistant
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_data():
    """加载原始CSV数据"""
    csv_path = RAW_DIR / "Perovskite_database_content_all_data.csv"
    print(f"📂 加载数据: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")

    return df

def clean_numeric_columns(df):
    """清洗数值列"""
    print("\n🔧 清洗数值列...")

    # 性能相关列
    performance_cols = [
        'JV_default_PCE', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF',
        'JV_reverse_scan_PCE', 'JV_reverse_scan_Voc', 'JV_reverse_scan_Jsc', 'JV_reverse_scan_FF',
        'JV_forward_scan_PCE', 'JV_forward_scan_Voc', 'JV_forward_scan_Jsc', 'JV_forward_scan_FF',
        'Perovskite_band_gap', 'Perovskite_thickness',
        'Cell_area_total', 'Cell_area_measured',
        'Stability_PCE_initial_value', 'Stability_PCE_end_of_experiment'
    ]

    for col in performance_cols:
        if col in df.columns:
            # 处理字符串类型的数值
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # 过滤掉 PCE 为 0 或负数的记录
    if 'JV_default_PCE' in df.columns:
        df = df[df['JV_default_PCE'] > 0].copy()

    return df

def clean_categorical_columns(df):
    """清洗分类列"""
    print("\n🔧 清洗分类列...")

    # 钙钛矿材料分类
    material_cols = [
        'Perovskite_composition_a_ions',
        'Perovskite_composition_b_ions',
        'Perovskite_composition_c_ions',
        'Perovskite_composition_short_form'
    ]

    for col in material_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # 架构分类
    if 'Cell_architecture' in df.columns:
        df['Cell_architecture'] = df['Cell_architecture'].fillna('Unknown')

    return df

def extract_features(df):
    """提取特征"""
    print("\n🔬 提取特征...")

    # 1. 钙钛矿类型分类
    def classify_perovskite(row):
        a_ions = str(row.get('Perovskite_composition_a_ions', '')).upper()
        b_ions = str(row.get('Perovskite_composition_b_ions', '')).upper()

        # A位离子
        if 'MA' in a_ions and 'FA' in a_ions:
            a_type = 'MAFA'
        elif 'MA' in a_ions and 'CS' in a_ions:
            a_type = 'MACs'
        elif 'FA' in a_ions and 'CS' in a_ions:
            a_type = 'FACs'
        elif 'MA' in a_ions and 'FA' in a_ions and 'CS' in a_ions:
            a_type = 'Triple'
        elif 'MA' in a_ions:
            a_type = 'MA'
        elif 'FA' in a_ions:
            a_type = 'FA'
        elif 'CS' in a_ions:
            a_type = 'Cs'
        else:
            a_type = 'Other'

        # B位离子
        if 'SN' in b_ions:
            b_type = 'Sn'
        elif 'GE' in b_ions:
            b_type = 'Ge'
        elif 'PB' in b_ions:
            b_type = 'Pb'
        else:
            b_type = 'Other'

        return f"{a_type}-{b_type}"

    df['perovskite_type'] = df.apply(classify_perovskite, axis=1)

    # 2. 是否无铅
    df['is_lead_free'] = df['Perovskite_composition_leadfree'].fillna(False).astype(bool)

    # 3. 是否无机
    df['is_inorganic'] = df['Perovskite_composition_inorganic'].fillna(False).astype(bool)

    # 4. 维度类型
    df['dimension_type'] = '3D'
    df.loc[df['Perovskite_dimension_2D'].fillna(False).astype(bool), 'dimension_type'] = '2D'
    df.loc[df['Perovskite_dimension_2D3D_mixture'].fillna(False).astype(bool), 'dimension_type'] = '2D/3D'
    df.loc[df['Perovskite_dimension_0D'].fillna(False).astype(bool), 'dimension_type'] = '0D'

    return df

def calculate_derived_features(df):
    """计算派生特征"""
    print("\n🧮 计算派生特征...")

    # 滞后指数
    if all(col in df.columns for col in ['JV_reverse_scan_PCE', 'JV_forward_scan_PCE']):
        df['hysteresis_index'] = (
            (df['JV_reverse_scan_PCE'] - df['JV_forward_scan_PCE']) /
            df['JV_reverse_scan_PCE'].replace(0, np.nan)
        ).abs()

    # 效率分类
    if 'JV_default_PCE' in df.columns:
        df['efficiency_class'] = pd.cut(
            df['JV_default_PCE'],
            bins=[0, 10, 15, 20, 25, 100],
            labels=['<10%', '10-15%', '15-20%', '20-25%', '>25%']
        )

    return df

def save_processed_data(df):
    """保存处理后的数据"""
    print("\n💾 保存处理后的数据...")

    # 保存完整数据
    output_path = PROCESSED_DIR / "perovskite_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"   ✅ 完整数据: {output_path}")

    # 保存用于 ML 的数据集
    ml_cols = [
        'Ref_ID', 'Ref_publication_date', 'Ref_journal',
        'Cell_architecture', 'perovskite_type', 'dimension_type',
        'is_lead_free', 'is_inorganic',
        'Perovskite_band_gap', 'Perovskite_thickness',
        'JV_default_PCE', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF',
        'JV_reverse_scan_PCE', 'JV_forward_scan_PCE',
        'hysteresis_index', 'efficiency_class'
    ]

    ml_cols_available = [col for col in ml_cols if col in df.columns]
    df_ml = df[ml_cols_available].dropna(subset=['JV_default_PCE'])

    ml_path = PROCESSED_DIR / "perovskite_ml_ready.csv"
    df_ml.to_csv(ml_path, index=False)
    print(f"   ✅ ML 数据: {ml_path} ({len(df_ml)} 行)")

    return df

def generate_data_report(df):
    """生成数据报告"""
    print("\n📝 生成数据报告...")

    report = f"""# 钙钛矿数据库数据清洗报告

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 数据概览

| 指标 | 值 |
|------|-----|
| 总记录数 | {len(df):,} |
| 字段数 | {len(df.columns)} |

## 🔬 性能数据统计

### PCE (光电转换效率)
- 有效样本数: {df['JV_default_PCE'].notna().sum():,}
- 平均值: {df['JV_default_PCE'].mean():.2f}%
- 最大值: {df['JV_default_PCE'].max():.2f}%
- 最小值: {df['JV_default_PCE'].min():.2f}%
- 标准差: {df['JV_default_PCE'].std():.2f}%

### Voc (开路电压)
- 有效样本数: {df['JV_default_Voc'].notna().sum():,}
- 平均值: {df['JV_default_Voc'].mean():.3f} V

### Jsc (短路电流)
- 有效样本数: {df['JV_default_Jsc'].notna().sum():,}
- 平均值: {df['JV_default_Jsc'].mean():.2f} mA/cm²

### FF (填充因子)
- 有效样本数: {df['JV_default_FF'].notna().sum():,}
- 平均值: {df['JV_default_FF'].mean():.2f}%

## 🧪 材料分布

### 钙钛矿类型
```
{df['perovskite_type'].value_counts().head(10).to_string()}
```

### 电池架构
```
{df['Cell_architecture'].value_counts().head(10).to_string()}
```

### 维度类型
```
{df['dimension_type'].value_counts().to_string()}
```

### 无铅/无机比例
- 无铅电池: {df['is_lead_free'].sum():,} ({df['is_lead_free'].mean()*100:.1f}%)
- 无机钙钛矿: {df['is_inorganic'].sum():,} ({df['is_inorganic'].mean()*100:.1f}%)

---

**数据来源**: Jesperkemist/perovskitedatabase_data
"""

    report_path = PROCESSED_DIR / "data_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"   ✅ 报告: {report_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("🧹 钙钛矿数据库 - 数据清洗")
    print("=" * 60)

    # 加载数据
    df = load_raw_data()

    # 清洗数据
    df = clean_numeric_columns(df)
    df = clean_categorical_columns(df)

    # 提取特征
    df = extract_features(df)
    df = calculate_derived_features(df)

    # 保存数据
    df = save_processed_data(df)

    # 生成报告
    generate_data_report(df)

    print("\n" + "=" * 60)
    print("✅ 数据清洗完成！")
    print("=" * 60)
    print(f"\n📁 输出目录: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
