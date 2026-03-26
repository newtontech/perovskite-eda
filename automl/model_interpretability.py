#!/usr/bin/env python3
"""
模型可解释性分析模块
Model Interpretability Analysis using SHAP and LIME

作者: OpenClaw AI Assistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports" / "interpretability"
FIGURES_DIR = PROJECT_ROOT / "figures" / "interpretability"

# 创建目录
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class ModelInterpretability:
    """模型可解释性分析类"""

    def __init__(self, model, feature_names, class_names=None):
        """
        初始化

        Args:
            model: 训练好的模型 (sklearn compatible)
            feature_names: 特征名称列表
            class_names: 类别名称 (可选)
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None
        self.shap_values = None
        self.lime_explainer = None

    def compute_shap_values(self, X, algorithm='auto'):
        """
        计算 SHAP 值

        Args:
            X: 特征数据 (DataFrame or ndarray)
            algorithm: 'auto', 'tree', 'linear', 'kernel'
        """
        print("\n🔍 计算 SHAP 值...")

        # 选择算法
        if algorithm == 'auto':
            if hasattr(self.model, 'predict_proba'):
                algorithm = 'tree'
            else:
                algorithm = 'kernel'

        # 创建解释器
        if hasattr(self.model, 'predict_proba'):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict, X)

        # 计算 SHAP 值
        self.shap_values = self.explainer.shap_values(X)

        print(f"   ✅ SHAP 值计算完成: {self.shap_values.shape}")
        return self.shap_values

    def plot_shap_summary(self, X, max_display=20, save=True):
        """
        绘制 SHAP Summary Plot

        Args:
            X: 特征数据
            max_display: 显示的最大特征数
            save: 是否保存图片
        """
        print("\n📊 绘制 SHAP Summary Plot...")

        fig, ax = plt.subplots(figsize=(12, 10))

        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = self.feature_names

        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )

        plt.tight_layout()

        if save:
            plt.savefig(
                FIGURES_DIR / "shap_summary.png",
                dpi=300,
                bbox_inches='tight'
            )
            print(f"   ✅ 保存: figures/interpretability/shap_summary.png")

        plt.close()

    def plot_shap_dependence(self, X, feature_idx, interaction_feature=None, save=True):
        """
        绘制 SHAP Dependence Plot

        Args:
            X: 特征数据
            feature_idx: 特征索引或名称
            interaction_feature: 交互特征
            save: 是否保存图片
        """
        print(f"\n📊 绘制 SHAP Dependence Plot for {feature_idx}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = self.feature_names

        if isinstance(feature_idx, int):
            feature_name = feature_names[feature_idx]
        else:
            feature_name = feature_idx
            feature_idx = feature_names.index(feature_idx)

        shap.dependence_plot(
            feature_name,
            self.shap_values,
            X,
            feature_names=feature_names,
            interaction_index=interaction_feature,
            show=False
        )

        plt.tight_layout()

        if save:
            plt.savefig(
                FIGURES_DIR / f"shap_dependence_{feature_name}.png",
                dpi=300,
                bbox_inches='tight'
            )
            print(f"   ✅ 保存: figures/interpretability/shap_dependence_{feature_name}.png")

        plt.close()

    def compute_lime_explanation(self, X_instance, num_features=10):
        """
        计算 LIME 解释

        Args:
            X_instance: 单个样本 (Series or array)
            num_features: 显示的特征数

        Returns:
            LIME 解释对象
        """
        print("\n🔍 计算 LIME 解释...")

        # 创建 LIME 解释器
        if self.lime_explainer is None:
            if isinstance(X_instance, pd.Series):
                X_df = X_instance.to_frame().T
            else:
                X_df = X_instance.reshape(1, -1)

            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_df.values,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='regression'
            )

        # 获取解释
        if isinstance(X_instance, pd.Series):
            X_exp = X_instance.values
        else:
            X_exp = X_instance

        explanation = self.lime_explainer.explain_instance(
            X_exp,
            self.model.predict,
            num_features=num_features
        )

        print("   ✅ LIME 解释计算完成")
        return explanation

    def plot_lime_explanation(self, explanation, save=True):
        """
        绘制 LIME 解释

        Args:
            explanation: LIME 解释对象
            save: 是否保存图片
        """
        print("\n📊 绘制 LIME 解释...")

        fig = explanation.as_pyplot_figure()
        plt.tight_layout()

        if save:
            plt.savefig(
                FIGURES_DIR / "lime_explanation.png",
                dpi=300,
                bbox_inches='tight'
            )
            print(f"   ✅ 保存: figures/interpretability/lime_explanation.png")

        plt.close()

    def get_feature_importance(self, X, method='shap', top_k=15):
        """
        获取特征重要性

        Args:
            X: 特征数据
            method: 'shap' or 'permutation'
            top_k: 返回前 k 个重要特征

        Returns:
            DataFrame: 特征重要性排名
        """
        print(f"\n📈 计算特征重要性 ({method})...")

        if method == 'shap':
            if self.shap_values is None:
                self.compute_shap_values(X)

            # 计算平均绝对 SHAP 值
            if len(self.shap_values.shape) == 2:
                importance = np.abs(self.shap_values).mean(axis=0)
            else:
                importance = np.abs(self.shap_values).mean(axis=0)

            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

        elif method == 'permutation':
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                self.model, X, n_repeats=10, random_state=42
            )

            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': result.importances_mean,
                'std': result.importances_std
            }).sort_values('importance', ascending=False)

        importance_df = importance_df.head(top_k)
        print(f"   ✅ 特征重要性计算完成 (Top {top_k})")

        return importance_df

    def generate_report(self, feature_importance_df):
        """生成可解释性报告"""
        print("\n📝 生成可解释性报告...")

        report = f"""# 模型可解释性分析报告

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据来源**: Perovskite Database

---

## 🔍 特征重要性排名 (SHAP)

| 排名 | 特征 | 重要性 |
|------|------|--------|
"""

        for i, (_, row) in enumerate(feature_importance_df.iterrows(), 1):
            report += f"| {i} | {row['feature']} | {row['importance']:.4f} |\n"

        report += f"""
---

## 📊 SHAP 分析解读

### 最重要的特征

1. **{feature_importance_df.iloc[0]['feature']}**: 重要性 {feature_importance_df.iloc[0]['importance']:.4f}
   - 对预测结果影响最大

2. **{feature_importance_df.iloc[1]['feature']}**: 重要性 {feature_importance_df.iloc[1]['importance']:.4f}
   - 第二重要特征

3. **{feature_importance_df.iloc[2]['feature']}**: 重要性 {feature_importance_df.iloc[2]['importance']:.4f}
   - 第三重要特征

---

## 📈 可视化图表

- `figures/interpretability/shap_summary.png` - SHAP 摘要图
- `figures/interpretability/shap_dependence_*.png` - SHAP 依赖图
- `figures/interpretability/lime_explanation.png` - LIME 解释图

---

## 🔬 科学洞察

基于 SHAP 分析，以下是影响钙钛矿太阳能电池 PCE 的关键因素：

"""

        # 添加科学洞察
        top_features = feature_importance_df['feature'].head(5).tolist()
        for i, feat in enumerate(top_features, 1):
            report += f"{i}. **{feat}**\n"

        report += """
---

**分析工具**: OpenClaw AI Assistant
**方法**: SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanations)
"""

        report_path = REPORTS_DIR / "interpretability_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"   ✅ 报告: {report_path}")
        return report_path


def run_interpretability_analysis():
    """运行完整的可解释性分析"""
    print("=" * 60)
    print("🔍 模型可解释性分析")
    print("=" * 60)

    # 加载数据
    print("\n📂 加载数据...")
    processed_path = DATA_DIR / "processed" / "perovskite_cleaned.csv"
    raw_path = DATA_DIR / "raw" / "Perovskite_database_content_all_data.csv"

    if processed_path.exists():
        df = pd.read_csv(processed_path)
        print(f"   加载处理后数据: {len(df)} 行")
    elif raw_path.exists():
        df = pd.read_csv(raw_path, low_memory=False)
        print(f"   加载原始数据: {len(df)} 行")
    else:
        raise FileNotFoundError("数据文件不存在")

    # 准备特征
    feature_cols = [
        'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF',
        'Perovskite_band_gap', 'Perovskite_thickness'
    ]

    # 检查可用的特征列
    available_cols = [col for col in feature_cols if col in df.columns]
    if len(available_cols) < 3:
        print("   ⚠️ 特征不足，使用默认特征列")
        available_cols = ['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF']

    # 准备数据
    X = df[available_cols].dropna()
    y = df.loc[X.index, 'JV_default_PCE'].dropna()
    X = X.loc[y.index]

    # 限制样本数
    max_samples = 5000
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)
        y = y.loc[X.index]

    print(f"   有效样本: {len(X)}")

    # 训练模型
    print("\n🤖 训练模型...")
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    print("   ✅ 模型训练完成")

    # 创建解释器
    interpreter = ModelInterpretability(
        model=model,
        feature_names=available_cols
    )

    # 计算 SHAP 值
    interpreter.compute_shap_values(X)

    # 绘制 SHAP Summary
    interpreter.plot_shap_summary(X)

    # 获取特征重要性
    importance_df = interpreter.get_feature_importance(X, method='shap')

    # 打印特征重要性
    print("\n📊 特征重要性:")
    print(importance_df.to_string(index=False))

    # 生成 LIME 解释 (对单个样本)
    instance = X.iloc[0]
    explanation = interpreter.compute_lime_explanation(instance)
    interpreter.plot_lime_explanation(explanation)

    # 生成报告
    interpreter.generate_report(importance_df)

    print("\n" + "=" * 60)
    print("✅ 可解释性分析完成！")
    print("=" * 60)
    print(f"\n📊 查看报告: {REPORTS_DIR / 'interpretability_report.md'}")
    print(f"📈 查看图表: {FIGURES_DIR}/")


if __name__ == "__main__":
    run_interpretability_analysis()