#!/usr/bin/env python3
"""
AutoML 分析：自动化机器学习预测钙钛矿性能
Automated Machine Learning for Perovskite Performance Prediction

基于: Jesperkemist/perovskitedatabase_data
作者: OpenClaw AI Assistant
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports" / "automl"
FIGURES_DIR = PROJECT_ROOT / "figures" / "automl"

for dir_path in [REPORTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class PerovskiteAutoML:
    """钙钛矿性能预测 AutoML 类"""

    def __init__(self):
        """初始化"""
        self.df = None
        self.best_model = None
        self.feature_importance = None
        self.results = None

    def load_data(self):
        """加载数据"""
        # 加载 ML 就绪的数据
        ml_path = DATA_DIR / "processed" / "perovskite_ml_ready.csv"
        cleaned_path = DATA_DIR / "processed" / "perovskite_cleaned.csv"
        raw_path = DATA_DIR / "raw" / "Perovskite_database_content_all_data.csv"

        if ml_path.exists():
            print(f"📂 加载 ML 数据: {ml_path}")
            self.df = pd.read_csv(ml_path)
        elif cleaned_path.exists():
            print(f"📂 加载清洗后数据: {cleaned_path}")
            self.df = pd.read_csv(cleaned_path)
        elif raw_path.exists():
            print(f"📂 加载原始数据: {raw_path}")
            self.df = pd.read_csv(raw_path, low_memory=False)
        else:
            raise FileNotFoundError("数据文件不存在！请先运行 clean_data.py")

        print(f"✅ 数据加载成功: {len(self.df):,} 行")

    def prepare_features(self):
        """准备特征"""
        print("\n🔧 准备特征...")

        # 定义特征列
        numerical_features = [
            'Perovskite_band_gap', 'Perovskite_thickness',
            'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF'
        ]

        categorical_features = [
            'Cell_architecture', 'perovskite_type', 'dimension_type'
        ]

        # 目标变量
        target = 'JV_default_PCE'

        # 检查可用列
        available_num = [col for col in numerical_features if col in self.df.columns]
        available_cat = [col for col in categorical_features if col in self.df.columns]

        print(f"   数值特征: {available_num}")
        print(f"   分类特征: {available_cat}")

        # 过滤有效数据
        required_cols = available_num + [target]
        df_ml = self.df.dropna(subset=required_cols).copy()

        # 处理分类变量
        from sklearn.preprocessing import LabelEncoder
        label_encoders = {}

        for col in available_cat:
            if col in df_ml.columns:
                le = LabelEncoder()
                df_ml[f'{col}_encoded'] = le.fit_transform(df_ml[col].astype(str))
                label_encoders[col] = le
                available_num.append(f'{col}_encoded')

        # 准备 X 和 y
        X = df_ml[available_num].copy()
        y = df_ml[target].copy()

        # 确保所有列都是数值类型
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        # 处理缺失值
        X = X.fillna(X.median(numeric_only=True))

        print(f"   最终特征数: {len(available_num)}")
        print(f"   有效样本数: {len(X):,}")

        self.features = available_num
        self.X = X
        self.y = y
        self.label_encoders = label_encoders

        return X, y

    def run_automl(self):
        """运行 AutoML"""
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        print("\n🤖 运行 AutoML...")

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # 定义模型
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        # 尝试加载 XGBoost 和 LightGBM
        try:
            import xgboost as xgb
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        except ImportError:
            print("   ⚠️ XGBoost 未安装，跳过")

        try:
            import lightgbm as lgb
            models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        except ImportError:
            print("   ⚠️ LightGBM 未安装，跳过")

        # 训练和评估
        results = []
        trained_models = {}

        for name, model in models.items():
            print(f"\n   训练 {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            results.append({
                'Model': name,
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae
            })

            trained_models[name] = model

            print(f"      R² Score: {r2:.4f}")
            print(f"      RMSE: {rmse:.4f}")
            print(f"      MAE: {mae:.4f}")

        # 保存结果
        self.results = pd.DataFrame(results).sort_values('R2', ascending=False)

        # 选择最佳模型
        best_model_name = self.results.iloc[0]['Model']
        self.best_model = trained_models[best_model_name]

        print(f"\n✅ 最佳模型: {best_model_name}")
        print(f"   R² Score: {self.results.iloc[0]['R2']:.4f}")

        # 绘制模型对比图
        self._plot_model_comparison()

        return self.results

    def _plot_model_comparison(self):
        """绘制模型对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # R² 对比
        ax = axes[0]
        colors = ['steelblue' if i > 0 else 'coral' for i in range(len(self.results))]
        ax.barh(self.results['Model'], self.results['R2'], color=colors)
        ax.set_xlabel('R² Score')
        ax.set_title('Model R² Comparison')
        ax.grid(axis='x', alpha=0.3)

        # RMSE 对比
        ax = axes[1]
        ax.barh(self.results['Model'], self.results['RMSE'], color=colors)
        ax.set_xlabel('RMSE')
        ax.set_title('Model RMSE Comparison')
        ax.grid(axis='x', alpha=0.3)

        # MAE 对比
        ax = axes[2]
        ax.barh(self.results['Model'], self.results['MAE'], color=colors)
        ax.set_xlabel('MAE')
        ax.set_title('Model MAE Comparison')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n   ✅ 模型对比图: {FIGURES_DIR / 'model_comparison.png'}")

    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n🔍 分析特征重要性...")

        if self.best_model is None:
            print("⚠️ 请先运行 AutoML")
            return

        # 获取特征重要性
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_

            self.feature_importance = pd.DataFrame({
                'Feature': self.features,
                'Importance': importance
            }).sort_values('Importance', ascending=True)

            print("\n特征重要性排序:")
            print(self.feature_importance.sort_values('Importance', ascending=False).to_string(index=False))

            # 绘制特征重要性图
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(self.feature_importance)))
            ax.barh(self.feature_importance['Feature'], self.feature_importance['Importance'], color=colors)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n   ✅ 特征重要性图: {FIGURES_DIR / 'feature_importance.png'}")

            # 保存 CSV
            self.feature_importance.sort_values('Importance', ascending=False).to_csv(
                REPORTS_DIR / 'feature_importance.csv', index=False
            )
        else:
            print("⚠️ 当前模型不支持特征重要性分析")

    def predict_new_material(self, sample_data=None):
        """预测新材料性能"""
        print("\n🔮 预测新材料性能...")

        if self.best_model is None:
            print("⚠️ 请先运行 AutoML")
            return

        if sample_data is None:
            # 使用测试集的样本
            sample = self.X.iloc[:5].copy()
        else:
            sample = pd.DataFrame([sample_data])

        predictions = self.best_model.predict(sample)

        print("\n预测结果:")
        for i, pred in enumerate(predictions):
            print(f"   样本 {i+1}: 预测 PCE = {pred:.2f}%")

        return predictions

    def generate_report(self):
        """生成 AutoML 报告"""
        print("\n📝 生成 AutoML 报告...")

        best_r2 = self.results.iloc[0]['R2']
        best_rmse = self.results.iloc[0]['RMSE']
        best_mae = self.results.iloc[0]['MAE']
        best_model = self.results.iloc[0]['Model']

        # 特征重要性表格
        if self.feature_importance is not None:
            fi_df = self.feature_importance.sort_values('Importance', ascending=False).head(10)
            fi_table = "| Feature | Importance |\n|---------|------------|\n"
            for _, row in fi_df.iterrows():
                fi_table += f"| {row['Feature']} | {row['Importance']:.4f} |\n"
            top_feature = self.feature_importance.iloc[-1]['Feature']
        else:
            fi_table = "暂无数据"
            top_feature = "N/A"

        report = f"""# AutoML 分析报告

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据来源**: Jesperkemist/perovskitedatabase_data
**样本数量**: {len(self.X):,}

---

## 📊 模型性能对比

| 模型 | R² | RMSE | MAE |
|------|-----|------|-----|
"""
        for _, row in self.results.iterrows():
            report += f"| {row['Model']} | {row['R2']:.4f} | {row['RMSE']:.4f} | {row['MAE']:.4f} |\n"

        report += f"""

### 最佳模型
- **模型类型**: {best_model}
- **R² Score**: {best_r2:.4f}
- **RMSE**: {best_rmse:.4f}
- **MAE**: {best_mae:.4f}

---

## 🔍 特征重要性分析 (Top 10)

{fi_table}

### 关键发现

1. **最重要特征**: {top_feature}
2. **模型解释力**: R² = {best_r2:.2%} 的方差可以被模型解释
3. **预测精度**: 平均绝对误差 (MAE) = {best_mae:.2f}%

---

## 📈 可视化图表

- `figures/automl/model_comparison.png` - 模型性能对比
- `figures/automl/feature_importance.png` - 特征重要性

---

## 🚀 优化建议

### 提高效率的关键因素
1. 优化高重要性特征的制备工艺
2. 重点关注特征之间的相互作用
3. 收集更多高重要性特征的数据

### 模型改进方向
1. 增加更多材料特征（如晶体结构、缺陷密度等）
2. 尝试深度学习模型
3. 进行超参数调优

---

**分析工具**: OpenClaw AI Assistant
"""

        report_path = REPORTS_DIR / "automl_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"   ✅ 报告: {report_path}")

    def run(self):
        """运行完整 AutoML 流程"""
        print("=" * 60)
        print("🤖 钙钛矿性能预测 - AutoML 分析")
        print("=" * 60)

        self.load_data()
        self.prepare_features()
        self.run_automl()
        self.analyze_feature_importance()
        self.predict_new_material()
        self.generate_report()

        print("\n" + "=" * 60)
        print("✅ AutoML 分析完成！")
        print("=" * 60)
        print(f"\n📊 查看报告: {REPORTS_DIR / 'automl_report.md'}")
        print(f"📈 查看图表: {FIGURES_DIR}/")


if __name__ == "__main__":
    automl = PerovskiteAutoML()
    automl.run()
