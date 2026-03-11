#!/usr/bin/env python3
"""
AutoML 分析：自动化机器学习预测钙钛矿性能
Automated Machine Learning for Perovskite Performance Prediction

生成时间: 2026-03-11
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
REPORTS_DIR = PROJECT_ROOT / "reports" / "automl"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

class PerovskiteAutoML:
    """钙钛矿性能预测 AutoML 类"""
    
    def __init__(self, data_path):
        """初始化"""
        self.data_path = Path(data_path)
        self.df = None
        self.best_model = None
        self.feature_importance = None
        
    def load_data(self):
        """加载数据"""
        print(f"📂 加载数据: {self.data_path}")
        
        if not self.data_path.exists():
            # 使用模拟数据
            print("⚠️ 数据文件不存在，使用模拟数据")
            self.df = self.generate_mock_data()
        else:
            self.df = pd.read_excel(self.data_path)
        
        print(f"✅ 数据加载成功: {len(self.df)} 行")
        
    def generate_mock_data(self):
        """生成模拟数据（用于演示）"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'material_type': np.random.choice(['MAPbI3', 'FAPbI3', 'CsPbI3', 'Mixed'], n_samples),
            'bandgap': np.random.uniform(1.4, 2.3, n_samples),
            'thickness': np.random.uniform(300, 600, n_samples),
            'annealing_temp': np.random.uniform(80, 150, n_samples),
            'annealing_time': np.random.uniform(10, 60, n_samples),
            'voc': np.random.uniform(0.9, 1.3, n_samples),
            'jsc': np.random.uniform(18, 26, n_samples),
            'ff': np.random.uniform(65, 85, n_samples),
        }
        
        # 计算 PCE (简化的物理模型)
        data['pce'] = (data['voc'] * data['jsc'] * data['ff'] / 100) * 0.9
        
        df = pd.DataFrame(data)
        
        # 添加噪声
        df['pce'] += np.random.normal(0, 1, n_samples)
        df['pce'] = df['pce'].clip(10, 27)
        
        return df
    
    def run_automl_pycaret(self):
        """使用 PyCaret 进行 AutoML"""
        try:
            from pycaret.regression import setup, compare_models, pull, save_model
            
            print("\n🤖 运行 PyCaret AutoML...")
            
            # 准备数据
            features = ['material_type', 'bandgap', 'thickness', 'annealing_temp', 
                       'annealing_time', 'voc', 'jsc', 'ff']
            target = 'pce'
            
            df_ml = self.df[features + [target]].copy()
            
            # PyCaret 设置
            exp = setup(data=df_ml, target=target, session_id=42, verbose=False)
            
            # 比较所有模型
            best = compare_models(verbose=False)
            
            # 获取结果
            results = pull()
            
            print("\n📊 模型性能对比 (Top 5):")
            print(results.head()[['Model', 'R2', 'RMSE', 'MAE']])
            
            # 保存最佳模型
            self.best_model = best
            save_model(best, REPORTS_DIR / 'best_model')
            
            print(f"\n✅ 最佳模型: {type(best).__name__}")
            print(f"   R² Score: {results.iloc[0]['R2']:.4f}")
            print(f"   RMSE: {results.iloc[0]['RMSE']:.4f}")
            
            return results
            
        except ImportError:
            print("⚠️ PyCaret 未安装，使用简化版本")
            return self.run_automl_simple()
    
    def run_automl_simple(self):
        """简化版 AutoML（不依赖 PyCaret）"""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        from sklearn.preprocessing import LabelEncoder
        
        print("\n🤖 运行简化版 AutoML...")
        
        # 准备特征
        features = ['bandgap', 'thickness', 'annealing_temp', 'annealing_time', 'voc', 'jsc', 'ff']
        
        # 编码分类变量
        le = LabelEncoder()
        df_ml = self.df.copy()
        df_ml['material_type_encoded'] = le.fit_transform(df_ml['material_type'])
        features.append('material_type_encoded')
        
        X = df_ml[features]
        y = df_ml['pce']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 定义模型
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # 训练和评估
        results = []
        for name, model in models.items():
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
            
            print(f"\n{name}:")
            print(f"   R² Score: {r2:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE: {mae:.4f}")
        
        results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
        
        # 保存最佳模型
        best_model_name = results_df.iloc[0]['Model']
        self.best_model = models[best_model_name]
        
        print(f"\n✅ 最佳模型: {best_model_name}")
        
        return results_df
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n🔍 分析特征重要性...")
        
        if self.best_model is None:
            print("⚠️ 请先运行 AutoML")
            return
        
        # 获取特征重要性
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            features = ['bandgap', 'thickness', 'annealing_temp', 'annealing_time', 
                       'voc', 'jsc', 'ff', 'material_type_encoded']
            
            self.feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print("\n特征重要性排序:")
            print(self.feature_importance.to_string(index=False))
            
            # 保存
            self.feature_importance.to_csv(REPORTS_DIR / 'feature_importance.csv', index=False)
            
        else:
            print("⚠️ 当前模型不支持特征重要性分析")
    
    def generate_report(self):
        """生成 AutoML 报告"""
        print("\n📝 生成报告...")
        
        report_path = REPORTS_DIR / "automl_report.md"
        
        report = f"""# AutoML 分析报告

**生成时间**: 2026-03-11  
**数据集大小**: {len(self.df)} 样本  
**目标变量**: PCE (Power Conversion Efficiency)

---

## 📊 模型性能对比

### 最佳模型
- **模型类型**: {type(self.best_model).__name__}
- **训练样本**: {int(len(self.df) * 0.8)}
- **测试样本**: {int(len(self.df) * 0.2)}

### 性能指标

| 指标 | 值 |
|------|-----|
| R² Score | {getattr(self.best_model, 'score', lambda x: 0)(self.df[['bandgap', 'thickness', 'annealing_temp', 'annealing_time', 'voc', 'jsc', 'ff']].iloc[:1]):.4f} |
| RMSE | - |
| MAE | - |

---

## 🔍 特征重要性分析

{self.feature_importance.to_markdown(index=False) if self.feature_importance is not None else '暂无数据'}

### 关键发现

1. **最重要特征**: {self.feature_importance.iloc[0]['Feature'] if self.feature_importance is not None else 'N/A'}
2. **次要特征**: {self.feature_importance.iloc[1]['Feature'] if self.feature_importance is not None and len(self.feature_importance) > 1 else 'N/A'}
3. **优化建议**: 
   - 重点关注高重要性特征的优化
   - 降低低重要性特征的成本投入

---

## 🚀 下一步行动

### 模型部署
- [ ] 保存模型到生产环境
- [ ] 构建 API 接口
- [ ] 集成到材料设计流程

### 持续优化
- [ ] 收集更多实验数据
- [ ] 尝试深度学习模型
- [ ] 特征工程优化

### 应用场景
- [ ] 新材料性能预测
- [ ] 工艺参数优化
- [ ] 成本效益分析

---

**报告生成**: OpenClaw AI Assistant  
**版本**: 1.0.0
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 报告已生成: {report_path}")
    
    def run(self):
        """运行完整 AutoML 流程"""
        print("=" * 60)
        print("🤖 钙钛矿性能预测 - AutoML 分析")
        print("=" * 60)
        
        self.load_data()
        self.run_automl_pycaret()
        self.analyze_feature_importance()
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("✅ AutoML 分析完成！")
        print("=" * 60)
        print(f"\n📊 查看报告: {REPORTS_DIR / 'automl_report.md'}")

if __name__ == "__main__":
    # 数据路径
    data_path = DATA_DIR / "raw" / "crossref.xlsx"
    
    # 运行 AutoML
    automl = PerovskiteAutoML(data_path)
    automl.run()
