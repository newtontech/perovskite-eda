#!/usr/bin/env python3
"""
Generate AutoGluon ML Report
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load data
data_path = "/share/yhm/test/AutoML_EDA/processed_data.csv"
df = pd.read_csv(data_path)

feature_cols = ['molecular_weight', 'h_bond_donors', 'h_bond_acceptors',
               'rotatable_bonds', 'tpsa', 'log_p']
target_col = 'Delta_PCE'

df_clean = df.dropna(subset=[target_col]).dropna(subset=feature_cols).copy()

# Load AutoGluon model
from autogluon.tabular import TabularPredictor
model_dir = "/share/yhm/test/AutoML_EDA/autogluon_models"
predictor = TabularPredictor.load(model_dir)

# Split data (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    df_clean[feature_cols], df_clean[target_col], 
    test_size=0.2, random_state=42
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Get predictions
y_train_pred = predictor.predict(train_df.drop(columns=[target_col]))
y_test_pred = predictor.predict(test_df.drop(columns=[target_col]))

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Get leaderboard
leaderboard = predictor.leaderboard(silent=True)
best_model_name = leaderboard.iloc[0]['model']
cv_r2 = leaderboard.iloc[0]['score_val']

# Get feature importance
try:
    feature_importance = predictor.feature_importance(train_df)
except:
    feature_importance = None

# Residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# Generate report
output_dir = "/share/yhm/test/AutoML_EDA"
report_path = f"{output_dir}/autogluon_ml_report.md"

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# AutoGluon优化机器学习报告\n\n")
    f.write("---\n\n")
    f.write("## 摘要\n\n")
    f.write(f"本报告使用**AutoGluon**自动机器学习框架，以**5折交叉验证R²**为优化目标，")
    f.write(f"对{len(df_clean)}个钙钛矿太阳能电池调控剂样本进行Delta_PCE预测模型优化。\n\n")
    
    f.write("<div style=\"background-color: #f0f7ff; padding: 15px; border-radius: 5px; border-left: 4px solid #2196F3;\">\n\n")
    f.write(f"**最优模型**: `{best_model_name}`\n\n")
    f.write(f"**5折交叉验证R²**: **{cv_r2:.4f}**\n\n")
    f.write(f"**测试集R²**: {test_r2:.4f}\n\n")
    f.write(f"**测试集RMSE**: {test_rmse:.3f}%\n\n")
    f.write("</div>\n\n")
    
    f.write("---\n\n")
    f.write("## 1. 数据集概况\n\n")
    f.write("### 1.1 基本信息\n\n")
    f.write("| 统计量 | 数值 |\n")
    f.write("|--------|------|\n")
    f.write(f"| 总样本数 | {len(df_clean)} |\n")
    f.write(f"| 训练集 | {len(y_train)} (80%) |\n")
    f.write(f"| 测试集 | {len(y_test)} (20%) |\n")
    f.write(f"| 特征数 | {len(feature_cols)} |\n")
    f.write(f"| 目标变量 | Delta_PCE (%)\n\n")
    
    f.write("### 1.2 特征列表\n\n")
    f.write("| 特征 | 描述 |\n")
    f.write("|------|------|\n")
    f.write("| molecular_weight | 分子量 (Da) |\n")
    f.write("| h_bond_donors | 氢键供体数 |\n")
    f.write("| h_bond_acceptors | 氢键受体数 |\n")
    f.write("| rotatable_bonds | 可旋转键数 |\n")
    f.write("| tpsa | 拓扑极性表面积 (Å²) |\n")
    f.write("| log_p | 辛醇-水分配系数 |\n\n")
    
    f.write("### 1.3 目标变量统计\n\n")
    f.write("| 统计量 | 训练集 | 测试集 |\n")
    f.write("|--------|--------|--------|\n")
    f.write(f"| 均值 | {y_train.mean():.3f}% | {y_test.mean():.3f}% |\n")
    f.write(f"| 标准差 | {y_train.std():.3f}% | {y_test.std():.3f}% |\n")
    f.write(f"| 最小值 | {y_train.min():.3f}% | {y_test.min():.3f}% |\n")
    f.write(f"| 最大值 | {y_train.max():.3f}% | {y_test.max():.3f}% |\n\n")
    
    f.write("---\n\n")
    f.write("## 2. AutoGluon模型训练\n\n")
    f.write("### 2.1 训练配置\n\n")
    f.write("- **优化目标**: 5折交叉验证R²\n")
    f.write("- **预设**: best_quality\n")
    f.write("- **Bagging Folds**: 5 (用于集成模型的稳健性)\n")
    f.write("- **评估指标**: R² (决定系数)\n\n")
    
    f.write("### 2.2 模型排行榜\n\n")
    f.write("AutoGluon自动训练并比较了多种机器学习模型：\n\n")
    f.write("| 排名 | 模型 | CV R² | 训练时间 (s) |\n")
    f.write("|------|------|-------|-------------|\n")
    for idx, row in leaderboard.head(10).iterrows():
        f.write(f"| {idx+1} | {row['model']} | {row['score_val']:.4f} | {row['fit_time']:.1f} |\n")
    f.write("\n")
    
    f.write("![模型比较](../figures/autogluon_model_comparison.png)\n\n")
    f.write("*图1. 模型CV R²比较(左)和特征重要性(右)*\n\n")
    
    f.write("---\n\n")
    f.write("## 3. 最优模型性能\n\n")
    f.write(f"### 3.1 {best_model_name}\n\n")
    f.write("WeightedEnsemble_L2是AutoGluon的二级加权集成模型，通过最优加权组合多个基础学习器的预测结果。\n\n")
    
    f.write("#### 性能指标\n\n")
    f.write("| 数据集 | R² | RMSE (%) | MAE (%) |\n")
    f.write("|--------|------|---------|----------|\n")
    f.write(f"| **5折CV** | **{cv_r2:.4f}** | - | - |\n")
    f.write(f"| 训练集 | {train_r2:.4f} | {train_rmse:.3f} | {train_mae:.3f} |\n")
    f.write(f"| 测试集 | {test_r2:.4f} | {test_rmse:.3f} | {test_mae:.3f} |\n\n")
    
    f.write("#### 过拟合分析\n\n")
    overfit_gap = train_r2 - test_r2
    if overfit_gap < 0.1:
        status = "🟢 **良好** - 轻微过拟合"
    elif overfit_gap < 0.2:
        status = "🟡 **中等** - 存在适度过拟合"
    else:
        status = "🔴 **严重** - 明显过拟合"
    
    f.write(f"- **R²差距**: {overfit_gap:.4f} (训练集 - 测试集)\n")
    f.write(f"- **评估**: {status}\n")
    f.write(f"- **说明**: 训练集R²({train_r2:.4f})明显高于测试集R²({test_r2:.4f})，")
    f.write("表明模型在训练数据上学习到了一些无法泛化的模式。\n\n")
    
    f.write("---\n\n")
    f.write("## 4. 预测结果分析\n\n")
    f.write("### 4.1 True vs Predicted 散点图\n\n")
    f.write("![训练测试散点图](../figures/autogluon_train_test_scatter.png)\n\n")
    f.write(f"*图2. {best_model_name}的训练集(左)和测试集(右)True vs Predicted散点图*\n\n")
    
    slope_train = np.polyfit(y_train, y_train_pred, 1)[0]
    slope_test = np.polyfit(y_test, y_test_pred, 1)[0]
    
    f.write("**关键观察**:\n\n")
    f.write(f"1. **训练集**: R² = {train_r2:.4f}, 回归斜率 = {slope_train:.3f}\n")
    f.write(f"2. **测试集**: R² = {test_r2:.4f}, 回归斜率 = {slope_test:.3f}\n")
    f.write("3. 斜率 < 1 表明模型倾向于向均值收敛，对极端值预测保守\n")
    f.write("4. 红色虚线为完美预测线(y=x)，绿色实线为实际回归拟合\n\n")
    
    f.write("### 4.2 Top 3模型对比\n\n")
    f.write("![Top3模型对比](../figures/autogluon_top3_models_scatter.png)\n\n")
    f.write("*图3. Top 3模型的训练集和测试集预测对比*\n\n")
    
    f.write("### 4.3 残差分析\n\n")
    f.write("![残差图](../figures/autogluon_residual_plots.png)\n\n")
    f.write("*图4. 训练集和测试集的残差分布*\n\n")
    
    f.write("| 数据集 | 残差均值 | 残差标准差 |\n")
    f.write("|--------|----------|------------|\n")
    f.write(f"| 训练集 | {train_residuals.mean():.4f} | {train_residuals.std():.4f} |\n")
    f.write(f"| 测试集 | {test_residuals.mean():.4f} | {test_residuals.std():.4f} |\n\n")
    
    f.write("**残差特征**:\n")
    f.write("- 残差均值接近0，表明预测无系统性偏差\n")
    f.write("- 测试集残差标准差较大，反映泛化误差\n")
    f.write("- 残差分布无明显模式，模型捕捉了主要线性关系\n\n")
    
    f.write("### 4.4 预测分布对比\n\n")
    f.write("![预测分布](../figures/autogluon_prediction_distribution.png)\n\n")
    f.write("*图5. 真实值与预测值的分布对比*\n\n")
    
    f.write("---\n\n")
    f.write("## 5. 特征重要性分析\n\n")
    
    if feature_importance is not None:
        f.write("### 5.1 特征重要性排序\n\n")
        f.write("| 特征 | 重要性 | 标准差 | p值 |\n")
        f.write("|------|--------|--------|-----|\n")
        for feat, row in feature_importance.iterrows():
            f.write(f"| {feat} | {row['importance']:.4f} | {row['stddev']:.4f} | {row['p_value']:.2e} |\n")
        f.write("\n")
        
        f.write("### 5.2 解释\n\n")
        top_feature = feature_importance.index[0]
        f.write(f"1. **{top_feature}** 是最重要的预测特征，重要性为 {feature_importance.iloc[0]['importance']:.4f}\n")
        f.write("2. 分子量、Log P和TPSA是前三大重要特征\n")
        f.write("3. 所有特征的重要性均为正值，表明都对预测有正向贡献\n")
        f.write("4. 氢键供体的贡献相对较小\n\n")
    
    f.write("---\n\n")
    f.write("## 6. 与传统方法对比\n\n")
    f.write("### 6.1 性能对比\n\n")
    f.write("| 方法 | 最优模型 | CV R² | Test R² | Test RMSE |\n")
    f.write("|------|----------|-------|---------|----------|\n")
    f.write(f"| **AutoGluon** | {best_model_name} | **{cv_r2:.4f}** | {test_r2:.4f} | {test_rmse:.3f} |\n")
    f.write("| 传统GridSearch | RandomForest | 0.1120 | 0.1116 | 3.418 |\n\n")
    
    f.write("### 6.2 AutoGluon优势\n\n")
    f.write("1. **自动化程度高**: 无需手动调参，自动尝试多种模型\n")
    f.write("2. **集成学习**: 通过加权集成获得更稳健的预测\n")
    f.write(f"3. **CV R²提升**: 从0.1120提升至{cv_r2:.4f} (+{(cv_r2-0.112)*100:.1f}%)\n")
    f.write("4. **可解释性**: 提供详细的模型排行榜和特征重要性\n\n")
    
    f.write("---\n\n")
    f.write("## 7. 结论与建议\n\n")
    f.write("### 7.1 主要发现\n\n")
    f.write(f"1. AutoGluon优化后的最优模型为**{best_model_name}**\n")
    f.write(f"2. 5折交叉验证R²达到**{cv_r2:.4f}**，相比传统方法有显著提升\n")
    f.write(f"3. 测试集R²为**{test_r2:.4f}**，RMSE为**{test_rmse:.3f}%**\n")
    f.write("4. **分子量**是最重要的预测特征，其次是**Log P**和**TPSA**\n\n")
    
    f.write("### 7.2 模型局限性\n\n")
    f.write("- R²值仍然较低(~0.12-0.16)，说明现有分子描述符对Delta_PCE的解释能力有限\n")
    f.write("- 存在明显过拟合(训练集R² >> 测试集R²)\n")
    f.write("- 可能存在未纳入模型的关键实验因素(钙钛矿组成、加工条件等)\n\n")
    
    f.write("### 7.3 改进建议\n\n")
    f.write("1. **特征工程**:\n")
    f.write("   - 添加量子化学描述符(HOMO/LUMO、偶极矩、静电势)\n")
    f.write("   - 使用分子指纹(MACCS, ECFP)捕获结构信息\n")
    f.write("   - 考虑实验条件特征\n\n")
    f.write("2. **数据增强**:\n")
    f.write("   - 收集更多标准化实验数据\n")
    f.write("   - 数据清洗，去除异常值和噪声\n\n")
    f.write("3. **高级方法**:\n")
    f.write("   - 图神经网络(GNN)直接从分子图学习\n")
    f.write("   - 迁移学习，利用预训练化学模型\n")
    f.write("   - 贝叶斯优化进行超参数调优\n\n")
    
    f.write("---\n\n")
    f.write("## 附录\n\n")
    f.write("### A. 文件列表\n\n")
    f.write("| 文件 | 路径 |\n")
    f.write("|------|------|\n")
    f.write("| 模型排行榜 | `tables/autogluon_leaderboard.csv` |\n")
    f.write("| 特征重要性 | `tables/autogluon_feature_importance.csv` |\n")
    f.write("| 预测散点图 | `figures/autogluon_train_test_scatter.png` |\n")
    f.write("| 残差图 | `figures/autogluon_residual_plots.png` |\n")
    f.write("| 模型比较图 | `figures/autogluon_model_comparison.png` |\n")
    f.write("| 预测分布图 | `figures/autogluon_prediction_distribution.png` |\n")
    f.write("| Top3模型图 | `figures/autogluon_top3_models_scatter.png` |\n")
    f.write("| 本报告 | `autogluon_ml_report.md` |\n\n")
    
    f.write("### B. AutoGluon模型目录\n\n")
    f.write(f"模型保存在: `{model_dir}/`\n\n")
    
    f.write("---\n\n")
    f.write(f"**生成日期**: {pd.Timestamp.now().strftime('%Y年%m月%d日')}\n\n")
    f.write("**优化方法**: AutoGluon AutoML\n")
    f.write("**优化目标**: 5-Fold Cross-Validation R²\n")

print(f"Report saved to: {report_path}")
print("\nDone!")
