#!/usr/bin/env python3
"""
模型泛化能力分析模块
Cross-domain validation and transfer learning for model generalization

作者: OpenClaw AI Assistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports" / "generalization"
FIGURES_DIR = PROJECT_ROOT / "figures" / "generalization"

# 创建目录
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class GeneralizationAnalyzer:
    """模型泛化能力分析类"""

    def __init__(self):
        """初始化"""
        self.df = None
        self.results = {}

    def load_data(self):
        """加载数据"""
        print("\n📂 加载数据...")

        processed_path = DATA_DIR / "processed" / "perovskite_cleaned.csv"
        raw_path = DATA_DIR / "raw" / "Perovskite_database_content_all_data.csv"

        if processed_path.exists():
            self.df = pd.read_csv(processed_path)
            print(f"   加载处理后数据: {len(self.df)} 行")
        elif raw_path.exists():
            self.df = pd.read_csv(raw_path, low_memory=False)
            print(f"   加载原始数据: {len(self.df)} 行")

        # 提取年份
        if 'Ref_publication_date' in self.df.columns:
            self.df['year'] = pd.to_datetime(
                self.df['Ref_publication_date'], errors='coerce'
            ).dt.year

        print(f"   ✅ 数据加载完成")

    def temporal_split_analysis(self, split_year=2020):
        """
        时间分割分析: 训练集用老数据，测试集用新数据

        Args:
            split_year: 分割年份

        Returns:
            dict: 分析结果
        """
        print(f"\n📊 时间分割分析 (split year: {split_year})...")

        if 'year' not in self.df.columns:
            print("   ⚠️ 无年份数据")
            return None

        # 准备特征
        feature_cols = [
            'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF',
            'Perovskite_band_gap', 'Perovskite_thickness'
        ]
        available_cols = [col for col in feature_cols if col in self.df.columns]

        # 过滤有效数据
        df_valid = self.df[
            self.df['JV_default_PCE'].notna() &
            self.df[available_cols].notna().all(axis=1)
        ].copy()

        # 分割
        train_df = df_valid[df_valid['year'] < split_year]
        test_df = df_valid[df_valid['year'] >= split_year]

        print(f"   训练集: {len(train_df)} (pre-{split_year})")
        print(f"   测试集: {len(test_df)} ({split_year}-present)")

        X_train = train_df[available_cols]
        y_train = train_df['JV_default_PCE']
        X_test = test_df[available_cols]
        y_test = test_df['JV_default_PCE']

        # 训练模型
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 计算指标
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        results = {
            'split_year': split_year,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'generalization_gap': train_r2 - test_r2
        }

        self.results['temporal_split'] = results

        print(f"\n   📈 结果:")
        print(f"   训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.3f}")
        print(f"   测试集 R²: {test_r2:.4f}, RMSE: {test_rmse:.3f}")
        print(f"   泛化差距: {results['generalization_gap']:.4f}")

        return results

    def cross_year_analysis(self, years=None):
        """
        跨年份泛化分析

        Args:
            years: 要分析的年份列表

        Returns:
            DataFrame: 各年份的性能
        """
        print("\n📊 跨年份泛化分析...")

        if 'year' not in self.df.columns:
            print("   ⚠️ 无年份数据")
            return None

        feature_cols = [
            'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF',
            'Perovskite_band_gap', 'Perovskite_thickness'
        ]
        available_cols = [col for col in feature_cols if col in self.df.columns]

        if years is None:
            years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score

        results = []

        # 用 2015-2018 的数据训练
        train_years = [2015, 2016, 2017, 2018]
        train_df = self.df[
            self.df['year'].isin(train_years) &
            self.df['JV_default_PCE'].notna() &
            self.df[available_cols].notna().all(axis=1)
        ]

        if len(train_df) < 100:
            print("   ⚠️ 训练数据不足")
            return None

        X_train = train_df[available_cols]
        y_train = train_df['JV_default_PCE']

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 测试各年份
        for year in years:
            test_df = self.df[
                (self.df['year'] == year) &
                self.df['JV_default_PCE'].notna() &
                self.df[available_cols].notna().all(axis=1)
            ]

            if len(test_df) < 10:
                continue

            X_test = test_df[available_cols]
            y_test = test_df['JV_default_PCE']

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            results.append({
                'year': year,
                'n_samples': len(test_df),
                'r2': r2,
                'mean_pce': y_test.mean(),
                'max_pce': y_test.max()
            })

        results_df = pd.DataFrame(results)
        self.results['cross_year'] = results_df

        print(f"   ✅ 分析完成: {len(results_df)} 个年份")
        print(results_df.to_string(index=False))

        return results_df

    def transfer_learning_analysis(self, threshold=20):
        """
        迁移学习分析

        Args:
            threshold: 高PCE阈值 (%)

        Returns:
            dict: 迁移学习结果
        """
        print(f"\n🔄 迁移学习分析 (高PCE阈值: {threshold}%)...")

        feature_cols = [
            'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF',
            'Perovskite_band_gap', 'Perovskite_thickness'
        ]
        available_cols = [col for col in feature_cols if col in self.df.columns]

        # 过滤数据
        df_valid = self.df[
            self.df['JV_default_PCE'].notna() &
            self.df[available_cols].notna().all(axis=1)
        ].copy()

        # 分割高/低 PCE
        high_pce = df_valid[df_valid['JV_default_PCE'] >= threshold]
        low_pce = df_valid[df_valid['JV_default_PCE'] < threshold]

        print(f"   高PCE样本: {len(high_pce)}")
        print(f"   低PCE样本: {len(low_pce)}")

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score, mean_squared_error

        # 方案1: 直接在低PCE上训练
        X_low = low_pce[available_cols]
        y_low = low_pce['JV_default_PCE']

        model_direct = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )
        model_direct.fit(X_low, y_low)

        # 方案2: 预训练 + 微调
        X_high = high_pce[available_cols]
        y_high = high_pce['JV_default_PCE']

        model_pretrained = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )
        model_pretrained.fit(X_high, y_high)
        model_pretrained.fit(X_low, y_low)  # 微调

        # 评估
        y_pred_direct = model_direct.predict(X_low)
        y_pred_transfer = model_pretrained.predict(X_low)

        r2_direct = r2_score(y_low, y_pred_direct)
        r2_transfer = r2_score(y_low, y_pred_transfer)
        rmse_direct = np.sqrt(mean_squared_error(y_low, y_pred_direct))
        rmse_transfer = np.sqrt(mean_squared_error(y_low, y_pred_transfer))

        results = {
            'threshold': threshold,
            'high_pce_samples': len(high_pce),
            'low_pce_samples': len(low_pce),
            'r2_direct': r2_direct,
            'r2_transfer': r2_transfer,
            'rmse_direct': rmse_direct,
            'rmse_transfer': rmse_transfer,
            'improvement': r2_transfer - r2_direct
        }

        self.results['transfer_learning'] = results

        print(f"\n   📈 结果:")
        print(f"   直接训练 R²: {r2_direct:.4f}, RMSE: {rmse_direct:.3f}")
        print(f"   迁移学习 R²: {r2_transfer:.4f}, RMSE: {rmse_transfer:.3f}")
        print(f"   提升: {results['improvement']:.4f}")

        return results

    def plot_temporal_trends(self):
        """绘制时间趋势图"""
        if 'cross_year' not in self.results:
            return

        df = self.results['cross_year']

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df['year'], df['r2'], 'o-', color='steelblue',
                linewidth=2, markersize=8, label='R² Score')

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Model Generalization Over Years', fontsize=14)
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "temporal_generalization.png",
            dpi=300,
            bbox_inches='tight'
        )
        print(f"\n   ✅ 保存: figures/generalization/temporal_generalization.png")
        plt.close()

    def generate_report(self):
        """生成泛化能力报告"""
        print("\n📝 生成泛化能力报告...")

        report = f"""# 模型泛化能力分析报告

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据来源**: Perovskite Database

---

## 1. 时间分割分析

"""

        if 'temporal_split' in self.results:
            r = self.results['temporal_split']
            report += f"""
### 训练/测试分割 (split year: {r['split_year']})

| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| 样本数 | {r['train_size']} | {r['test_size']} |
| R² | {r['train_r2']:.4f} | {r['test_r2']:.4f} |
| RMSE | {r['train_rmse']:.3f} | {r['test_rmse']:.3f} |
| MAE | {r['train_mae']:.3f} | {r['test_mae']:.3f} |

**泛化差距**: {r['generalization_gap']:.4f}

"""

        if 'cross_year' in self.results:
            report += f"""
---

## 2. 跨年份泛化分析

| 年份 | 样本数 | R² | 平均PCE | 最高PCE |
|------|--------|-----|---------|---------|
"""
            for _, row in self.results['cross_year'].iterrows():
                report += f"| {int(row['year'])} | {row['n_samples']} | {row['r2']:.4f} | {row['mean_pce']:.2f}% | {row['max_pce']:.2f}% |\n"

        if 'transfer_learning' in self.results:
            r = self.results['transfer_learning']
            report += f"""
---

## 3. 迁移学习分析

### 高PCE阈值: {r['threshold']}%

| 方法 | R² | RMSE |
|------|-----|------|
| 直接训练 | {r['r2_direct']:.4f} | {r['rmse_direct']:.3f} |
| 迁移学习 | {r['r2_transfer']:.4f} | {r['rmse_transfer']:.3f} |

**提升**: {r['improvement']:.4f}

"""

        report += f"""
---

## 📈 可视化图表

- `figures/generalization/temporal_generalization.png` - 时间趋势图

---

## 🔬 关键发现

1. **泛化能力**: 模型在新技术数据上的表现
2. **时间衰减**: 随着年份增加，预测性能的变化
3. **迁移效果**: 从高PCE样本迁移到低PCE样本的效果

---

**分析工具**: OpenClaw AI Assistant
"""

        report_path = REPORTS_DIR / "generalization_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"   ✅ 报告: {report_path}")
        return report_path


def run_generalization_analysis():
    """运行完整的泛化能力分析"""
    print("=" * 60)
    print("🌍 模型泛化能力分析")
    print("=" * 60)

    analyzer = GeneralizationAnalyzer()
    analyzer.load_data()

    # 时间分割分析
    analyzer.temporal_split_analysis(split_year=2020)

    # 跨年份分析
    analyzer.cross_year_analysis()

    # 迁移学习分析
    analyzer.transfer_learning_analysis(threshold=20)

    # 绘图
    analyzer.plot_temporal_trends()

    # 生成报告
    analyzer.generate_report()

    print("\n" + "=" * 60)
    print("✅ 泛化能力分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_generalization_analysis()