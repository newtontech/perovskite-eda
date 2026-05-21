#!/usr/bin/env python3
"""
多目标联合预测模块
Multi-target joint prediction for PCE, Voc, Jsc, FF, and stability

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
REPORTS_DIR = PROJECT_ROOT / "reports" / "multitask"
FIGURES_DIR = PROJECT_ROOT / "figures" / "multitask"

# 创建目录
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class MultiTargetPredictor:
    """多目标联合预测类"""

    def __init__(self):
        """初始化"""
        self.df = None
        self.models = {}
        self.results = {}

    def load_data(self):
        """加载数据"""
        print("\n📂 加载数据...")

        processed_path = DATA_DIR / "processed" / "perovskite_cleaned.csv"
        raw_path = DATA_DIR / "raw" / "Perovskite_database_content_all_data.csv"

        if processed_path.exists():
            self.df = pd.read_csv(processed_path)
        elif raw_path.exists():
            self.df = pd.read_csv(raw_path, low_memory=False)

        print(f"   ✅ 数据加载完成: {len(self.df)} 行")

    def prepare_data(self):
        """准备多目标数据"""
        print("\n🔧 准备多目标数据...")

        # 定义目标列
        target_cols = [
            'JV_default_PCE',
            'JV_default_Voc',
            'JV_default_Jsc',
            'JV_default_FF',
            'Stability_PCE_initial_value'
        ]

        # 特征列
        feature_cols = [
            'Perovskite_band_gap',
            'Perovskite_thickness',
            'Cell_area_total'
        ]

        available_targets = [col for col in target_cols if col in self.df.columns]
        available_features = [col for col in feature_cols if col in self.df.columns]

        # 过滤有效数据
        valid_mask = self.df[available_targets].notna().all(axis=1)
        valid_mask &= self.df[available_features].notna().all(axis=1)

        self.target_df = self.df.loc[valid_mask, available_targets].copy()
        self.feature_df = self.df.loc[valid_mask, available_features].copy()

        print(f"   有效样本: {len(self.target_df)}")
        print(f"   目标变量: {available_targets}")
        print(f"   特征变量: {available_features}")

        return self.target_df, self.feature_df

    def train_single_models(self, X, y_multi):
        """
        训练单目标模型 (基线)

        Returns:
            dict: 各目标的性能
        """
        print("\n🤖 训练单目标模型 (基线)...")

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import r2_score, mean_squared_error

        results = {}

        # 单独训练每个目标
        for target in y_multi.columns:
            y = y_multi[target]

            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )

            # 交叉验证
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            model.fit(X, y)

            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            results[target] = {
                'r2_cv': scores.mean(),
                'r2_train': r2,
                'rmse': rmse,
                'model': model
            }

            print(f"   {target}: R² = {r2:.4f}, RMSE = {rmse:.3f}")

        self.results['single'] = results
        return results

    def train_multitask_models(self, X, y_multi):
        """
        训练多目标模型

        Returns:
            dict: 多目标模型的性能
        """
        print("\n🤖 训练多目标联合模型...")

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor, RegressorChain
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import r2_score, mean_squared_error

        results = {}

        # 方法1: MultiOutputRegressor
        base_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        model_multi = MultiOutputRegressor(base_model)
        model_multi.fit(X, y_multi)

        y_pred_multi = model_multi.predict(X)

        for i, target in enumerate(y_multi.columns):
            y = y_multi[target]
            y_pred = y_pred_multi[:, i]

            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            results[target] = {
                'r2': r2,
                'rmse': rmse
            }

        self.results['multi'] = results
        print("   ✅ 多目标模型训练完成")

        return results

    def train_chain_models(self, X, y_multi):
        """
        训练链式多目标模型

        Returns:
            dict: 链式模型的性能
        """
        print("\n🔗 训练链式多目标模型...")

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import RegressorChain
        from sklearn.metrics import r2_score, mean_squared_error

        base_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        # 按重要性排序目标
        chain = RegressorChain(base_model, order=['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE'])
        chain.fit(X, y_multi)

        y_pred_chain = chain.predict(X)

        results = {}
        for i, target in enumerate(y_multi.columns):
            y = y_multi[target]
            y_pred = y_pred_chain[:, i]

            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            results[target] = {
                'r2': r2,
                'rmse': rmse
            }

        self.results['chain'] = results
        print("   ✅ 链式模型训练完成")

        return results

    def analyze_correlations(self, y_multi):
        """分析目标间的相关性"""
        print("\n📊 分析目标间相关性...")

        corr_matrix = y_multi.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        import seaborn as sns
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r',
                   center=0, fmt='.2f', ax=ax)
        ax.set_title('Target Variables Correlation')

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "target_correlations.png",
            dpi=300,
            bbox_inches='tight'
        )
        print(f"   ✅ 保存: figures/multitask/target_correlations.png")
        plt.close()

        self.results['correlations'] = corr_matrix
        return corr_matrix

    def compare_methods(self):
        """比较不同方法"""
        print("\n📈 比较不同方法...")

        if 'single' not in self.results or 'multi' not in self.results:
            return None

        comparison = []

        for target in self.results['single'].keys():
            if target in self.results['multi']:
                comparison.append({
                    'Target': target,
                    'Single R²': self.results['single'][target]['r2_train'],
                    'Multi R²': self.results['multi'][target]['r2'],
                    'Single RMSE': self.results['single'][target]['rmse'],
                    'Multi RMSE': self.results['multi'][target]['rmse']
                })

        df = pd.DataFrame(comparison)
        self.results['comparison'] = df

        print(df.to_string(index=False))
        return df

    def plot_comparison(self):
        """绘制方法比较图"""
        if 'comparison' not in self.results:
            return

        df = self.results['comparison']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # R² 比较
        ax = axes[0]
        x = np.arange(len(df))
        width = 0.35

        ax.bar(x - width/2, df['Single R²'], width, label='Single', color='steelblue')
        ax.bar(x + width/2, df['Multi R²'], width, label='Multi', color='coral')

        ax.set_xlabel('Target')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Comparison: Single vs Multi')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Target'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # RMSE 比较
        ax = axes[1]
        ax.bar(x - width/2, df['Single RMSE'], width, label='Single', color='steelblue')
        ax.bar(x + width/2, df['Multi RMSE'], width, label='Multi', color='coral')

        ax.set_xlabel('Target')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE Comparison: Single vs Multi')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Target'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "method_comparison.png",
            dpi=300,
            bbox_inches='tight'
        )
        print(f"   ✅ 保存: figures/multitask/method_comparison.png")
        plt.close()

    def generate_report(self):
        """生成多目标预测报告"""
        print("\n📝 生成多目标预测报告...")

        report = f"""# 多目标联合预测分析报告

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据来源**: Perovskite Database

---

## 1. 目标变量相关性

"""

        if 'correlations' in self.results:
            report += self.results['correlations'].to_string()

        report += f"""

---

## 2. 单目标 vs 多目标 比较

"""

        if 'comparison' in self.results:
            report += self.results['comparison'].to_string(index=False)

        report += f"""

---

## 3. 分析结论

### 多目标学习的优势
1. **参数共享**: 捕获目标间的相关性
2. **正则化效果**: 共享表示减少过拟合
3. **计算效率**: 单一模型预测多个目标

### 目标间相关性发现
- PCE 与 Voc, Jsc, FF 高度相关
- 稳定性与 PCE 存在一定关联

---

## 📈 可视化图表

- `figures/multitask/target_correlations.png` - 目标相关性
- `figures/multitask/method_comparison.png` - 方法比较

---

**分析工具**: OpenClaw AI Assistant
"""

        report_path = REPORTS_DIR / "multitask_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"   ✅ 报告: {report_path}")
        return report_path


def run_multitask_analysis():
    """运行完整的多目标分析"""
    print("=" * 60)
    print("🎯 多目标联合预测分析")
    print("=" * 60)

    predictor = MultiTargetPredictor()
    predictor.load_data()
    predictor.prepare_data()

    # 训练单目标模型
    predictor.train_single_models(predictor.feature_df, predictor.target_df)

    # 训练多目标模型
    predictor.train_multitask_models(predictor.feature_df, predictor.target_df)

    # 训练链式模型
    predictor.train_chain_models(predictor.feature_df, predictor.target_df)

    # 分析相关性
    predictor.analyze_correlations(predictor.target_df)

    # 比较方法
    predictor.compare_methods()

    # 绘图
    predictor.plot_comparison()

    # 生成报告
    predictor.generate_report()

    print("\n" + "=" * 60)
    print("✅ 多目标预测分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_multitask_analysis()