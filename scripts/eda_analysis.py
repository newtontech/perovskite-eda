#!/usr/bin/env python3
"""
钙钛矿太阳能电池数据库 - 探索性数据分析 (EDA)
Exploratory Data Analysis for Perovskite Solar Cells Database

基于: Jesperkemist/perovskitedatabase_data
作者: OpenClaw AI Assistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 创建目录
for dir_path in [FIGURES_DIR / "temporal", FIGURES_DIR / "materials",
                 FIGURES_DIR / "performance", FIGURES_DIR / "correlations"]:
    dir_path.mkdir(parents=True, exist_ok=True)

class PerovskiteEDA:
    """钙钛矿数据库 EDA 分析类"""

    def __init__(self):
        """初始化"""
        self.df = None
        self.report_lines = []

    def load_data(self):
        """加载数据"""
        # 优先加载处理后的数据
        processed_path = DATA_DIR / "processed" / "perovskite_cleaned.csv"
        raw_path = DATA_DIR / "raw" / "Perovskite_database_content_all_data.csv"

        if processed_path.exists():
            print(f"📂 加载处理后的数据: {processed_path}")
            self.df = pd.read_csv(processed_path)
        elif raw_path.exists():
            print(f"📂 加载原始数据: {raw_path}")
            self.df = pd.read_csv(raw_path, low_memory=False)
        else:
            raise FileNotFoundError("数据文件不存在！请先运行 clean_data.py")

        print(f"✅ 数据加载成功: {len(self.df):,} 行, {len(self.df.columns)} 列")

    def performance_overview(self):
        """性能数据概览"""
        print("\n📊 性能数据概览...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # PCE 分布
        ax = axes[0, 0]
        pce_data = self.df['JV_default_PCE'].dropna()
        ax.hist(pce_data, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(pce_data.mean(), color='red', linestyle='--', label=f'Mean: {pce_data.mean():.2f}%')
        ax.set_xlabel('PCE (%)')
        ax.set_ylabel('Count')
        ax.set_title('PCE Distribution')
        ax.legend()

        # Voc 分布
        ax = axes[0, 1]
        voc_data = self.df['JV_default_Voc'].dropna()
        ax.hist(voc_data, bins=50, color='forestgreen', alpha=0.7, edgecolor='white')
        ax.axvline(voc_data.mean(), color='red', linestyle='--', label=f'Mean: {voc_data.mean():.3f} V')
        ax.set_xlabel('Voc (V)')
        ax.set_ylabel('Count')
        ax.set_title('Voc Distribution')
        ax.legend()

        # Jsc 分布
        ax = axes[1, 0]
        jsc_data = self.df['JV_default_Jsc'].dropna()
        ax.hist(jsc_data, bins=50, color='coral', alpha=0.7, edgecolor='white')
        ax.axvline(jsc_data.mean(), color='red', linestyle='--', label=f'Mean: {jsc_data.mean():.2f} mA/cm²')
        ax.set_xlabel('Jsc (mA/cm²)')
        ax.set_ylabel('Count')
        ax.set_title('Jsc Distribution')
        ax.legend()

        # FF 分布
        ax = axes[1, 1]
        ff_data = self.df['JV_default_FF'].dropna()
        ax.hist(ff_data, bins=50, color='mediumpurple', alpha=0.7, edgecolor='white')
        ax.axvline(ff_data.mean(), color='red', linestyle='--', label=f'Mean: {ff_data.mean():.2f}%')
        ax.set_xlabel('FF (%)')
        ax.set_ylabel('Count')
        ax.set_title('FF Distribution')
        ax.legend()

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "performance" / "performance_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 保存: figures/performance/performance_distributions.png")

    def temporal_analysis(self):
        """时间趋势分析"""
        print("\n📅 时间趋势分析...")

        if 'Ref_publication_date' not in self.df.columns:
            print("   ⚠️ 无发表日期数据")
            return

        # 提取年份
        self.df['year'] = pd.to_datetime(self.df['Ref_publication_date'], errors='coerce').dt.year
        year_counts = self.df['year'].dropna().astype(int).value_counts().sort_index()

        # 过滤合理年份
        year_counts = year_counts[(year_counts.index >= 2009) & (year_counts.index <= 2025)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 发表趋势
        ax = axes[0]
        ax.bar(year_counts.index, year_counts.values, color='steelblue', alpha=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Records')
        ax.set_title('Publication Trend of Perovskite Solar Cells')
        ax.grid(axis='y', alpha=0.3)

        # PCE 年度趋势
        ax = axes[1]
        yearly_pce = self.df.groupby('year')['JV_default_PCE'].agg(['mean', 'max'])
        yearly_pce = yearly_pce[(yearly_pce.index >= 2009) & (yearly_pce.index <= 2025)]

        ax.plot(yearly_pce.index, yearly_pce['mean'], 'o-', color='steelblue', label='Mean PCE', linewidth=2)
        ax.plot(yearly_pce.index, yearly_pce['max'], 's--', color='coral', label='Max PCE', linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('PCE (%)')
        ax.set_title('PCE Trend Over Years')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "temporal" / "temporal_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 保存: figures/temporal/temporal_trends.png")

    def material_analysis(self):
        """材料分析"""
        print("\n🧪 材料分析...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 钙钛矿类型分布
        ax = axes[0, 0]
        if 'perovskite_type' in self.df.columns:
            type_counts = self.df['perovskite_type'].value_counts().head(15)
            ax.barh(range(len(type_counts)), type_counts.values, color='steelblue', alpha=0.8)
            ax.set_yticks(range(len(type_counts)))
            ax.set_yticklabels(type_counts.index)
            ax.set_xlabel('Count')
            ax.set_title('Perovskite Type Distribution')
            ax.grid(axis='x', alpha=0.3)

        # 电池架构分布
        ax = axes[0, 1]
        if 'Cell_architecture' in self.df.columns:
            arch_counts = self.df['Cell_architecture'].value_counts().head(10)
            colors = plt.cm.Set3(np.linspace(0, 1, len(arch_counts)))
            ax.pie(arch_counts.values, labels=arch_counts.index, autopct='%1.1f%%', colors=colors)
            ax.set_title('Cell Architecture Distribution')

        # 维度类型分布
        ax = axes[1, 0]
        if 'dimension_type' in self.df.columns:
            dim_counts = self.df['dimension_type'].value_counts()
            colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
            ax.bar(dim_counts.index, dim_counts.values, color=colors[:len(dim_counts)], alpha=0.8)
            ax.set_xlabel('Dimension Type')
            ax.set_ylabel('Count')
            ax.set_title('Dimension Type Distribution')
            ax.grid(axis='y', alpha=0.3)

        # 无铅/无机统计
        ax = axes[1, 1]
        if 'is_lead_free' in self.df.columns:
            lead_free = self.df['is_lead_free'].sum()
            lead_based = len(self.df) - lead_free
            inorganic = self.df['is_inorganic'].sum() if 'is_inorganic' in self.df.columns else 0

            x = ['Lead-based', 'Lead-free', 'Inorganic']
            y = [lead_based, lead_free, inorganic]
            colors = ['#607D8B', '#4CAF50', '#2196F3']
            bars = ax.bar(x, y, color=colors, alpha=0.8)
            ax.set_ylabel('Count')
            ax.set_title('Material Type Distribution')
            ax.grid(axis='y', alpha=0.3)

            # 添加数值标签
            for bar, val in zip(bars, y):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                       f'{val:,}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "materials" / "material_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 保存: figures/materials/material_analysis.png")

    def performance_by_material(self):
        """按材料分析性能"""
        print("\n📈 按材料分析性能...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 按钙钛矿类型的 PCE
        ax = axes[0]
        if 'perovskite_type' in self.df.columns:
            type_pce = self.df.groupby('perovskite_type')['JV_default_PCE'].agg(['mean', 'std', 'count'])
            type_pce = type_pce[type_pce['count'] >= 50].sort_values('mean', ascending=False).head(15)

            ax.barh(range(len(type_pce)), type_pce['mean'],
                   xerr=type_pce['std'].fillna(0), color='steelblue', alpha=0.8, capsize=3)
            ax.set_yticks(range(len(type_pce)))
            ax.set_yticklabels(type_pce.index)
            ax.set_xlabel('Mean PCE (%)')
            ax.set_title('Mean PCE by Perovskite Type (min 50 samples)')
            ax.grid(axis='x', alpha=0.3)

        # 按电池架构的 PCE
        ax = axes[1]
        if 'Cell_architecture' in self.df.columns:
            arch_pce = self.df.groupby('Cell_architecture')['JV_default_PCE'].agg(['mean', 'std', 'count'])
            arch_pce = arch_pce[arch_pce['count'] >= 50].sort_values('mean', ascending=False)

            ax.barh(range(len(arch_pce)), arch_pce['mean'],
                   xerr=arch_pce['std'].fillna(0), color='coral', alpha=0.8, capsize=3)
            ax.set_yticks(range(len(arch_pce)))
            ax.set_yticklabels(arch_pce.index)
            ax.set_xlabel('Mean PCE (%)')
            ax.set_title('Mean PCE by Cell Architecture (min 50 samples)')
            ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "performance" / "performance_by_material.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 保存: figures/performance/performance_by_material.png")

    def correlation_analysis(self):
        """相关性分析"""
        print("\n🔗 相关性分析...")

        # 选择数值列
        numeric_cols = ['JV_default_PCE', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF',
                       'Perovskite_band_gap', 'Perovskite_thickness']

        available_cols = [col for col in numeric_cols if col in self.df.columns]
        corr_data = self.df[available_cols].copy()

        # 确保所有列都是数值类型
        for col in available_cols:
            corr_data[col] = pd.to_numeric(corr_data[col], errors='coerce')

        corr_data = corr_data.dropna()

        if len(corr_data) < 10:
            print("   ⚠️ 数据不足，跳过相关性分析")
            return

        # 计算相关矩阵
        corr_matrix = corr_data.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   fmt='.2f', square=True, ax=ax)
        ax.set_title('Performance Parameters Correlation')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "correlations" / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 保存: figures/correlations/correlation_matrix.png")

    def generate_report(self):
        """生成 EDA 报告"""
        print("\n📝 生成 EDA 报告...")

        pce_data = self.df['JV_default_PCE'].dropna()
        voc_data = self.df['JV_default_Voc'].dropna()
        jsc_data = self.df['JV_default_Jsc'].dropna()
        ff_data = self.df['JV_default_FF'].dropna()

        report = f"""# 钙钛矿太阳能电池数据库 - EDA 分析报告

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据来源**: Jesperkemist/perovskitedatabase_data
**样本数量**: {len(self.df):,}

---

## 📊 性能数据概览

### PCE (光电转换效率)
| 统计量 | 值 |
|--------|-----|
| 有效样本数 | {len(pce_data):,} |
| 平均值 | {pce_data.mean():.2f}% |
| 最大值 | {pce_data.max():.2f}% |
| 最小值 | {pce_data.min():.2f}% |
| 中位数 | {pce_data.median():.2f}% |
| 标准差 | {pce_data.std():.2f}% |

### Voc (开路电压)
| 统计量 | 值 |
|--------|-----|
| 有效样本数 | {len(voc_data):,} |
| 平均值 | {voc_data.mean():.3f} V |
| 最大值 | {voc_data.max():.3f} V |

### Jsc (短路电流)
| 统计量 | 值 |
|--------|-----|
| 有效样本数 | {len(jsc_data):,} |
| 平均值 | {jsc_data.mean():.2f} mA/cm² |
| 最大值 | {jsc_data.max():.2f} mA/cm² |

### FF (填充因子)
| 统计量 | 值 |
|--------|-----|
| 有效样本数 | {len(ff_data):,} |
| 平均值 | {ff_data.mean():.2f}% |
| 最大值 | {ff_data.max():.2f}% |

---

## 🧪 材料分布

### 钙钛矿类型 (Top 10)
```
{self.df['perovskite_type'].value_counts().head(10).to_string() if 'perovskite_type' in self.df.columns else 'N/A'}
```

### 电池架构 (Top 10)
```
{self.df['Cell_architecture'].value_counts().head(10).to_string() if 'Cell_architecture' in self.df.columns else 'N/A'}
```

### 维度类型
```
{self.df['dimension_type'].value_counts().to_string() if 'dimension_type' in self.df.columns else 'N/A'}
```

---

## 📈 可视化图表

- `figures/performance/performance_distributions.png` - 性能分布图
- `figures/temporal/temporal_trends.png` - 时间趋势图
- `figures/materials/material_analysis.png` - 材料分析图
- `figures/performance/performance_by_material.png` - 按材料性能分析
- `figures/correlations/correlation_matrix.png` - 相关性矩阵

---

## 🔍 关键发现

1. **最高效率**: {pce_data.max():.2f}% (光电转换效率)
2. **主流材料**: {self.df['perovskite_type'].value_counts().index[0] if 'perovskite_type' in self.df.columns else 'N/A'}
3. **主流架构**: {self.df['Cell_architecture'].value_counts().index[0] if 'Cell_architecture' in self.df.columns else 'N/A'}

---

**分析工具**: OpenClaw AI Assistant
"""

        report_path = REPORTS_DIR / "eda_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"   ✅ 报告: {report_path}")

    def run(self):
        """运行完整分析"""
        print("=" * 60)
        print("🔬 钙钛矿太阳能电池数据库 - EDA 分析")
        print("=" * 60)

        self.load_data()
        self.performance_overview()
        self.temporal_analysis()
        self.material_analysis()
        self.performance_by_material()
        self.correlation_analysis()
        self.generate_report()

        print("\n" + "=" * 60)
        print("✅ EDA 分析完成！")
        print("=" * 60)
        print(f"\n📊 查看报告: {REPORTS_DIR / 'eda_report.md'}")
        print(f"📈 查看图表: {FIGURES_DIR}/")


if __name__ == "__main__":
    eda = PerovskiteEDA()
    eda.run()
