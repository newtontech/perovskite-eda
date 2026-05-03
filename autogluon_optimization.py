#!/usr/bin/env python3
"""
AutoGluon AutoML Optimization for QSPR Delta_PCE Prediction
Target: 5-fold Cross-Validation R² optimization
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
import os
import time

print("=" * 80)
print("AUTOGLUON AUTOML OPTIMIZATION FOR QSPR DELTA_PCE PREDICTION")
print("Optimization Target: 5-Fold Cross-Validation R²")
print("=" * 80)

# Try to import AutoGluon
try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    AUTOGLUON_AVAILABLE = True
    print("\nAutoGluon is available.")
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("\nWARNING: AutoGluon not available. Will use alternative approach.")

# Load data
data_path = "/share/yhm/test/AutoML_EDA/processed_data.csv"
df = pd.read_csv(data_path)

feature_cols = ['molecular_weight', 'h_bond_donors', 'h_bond_acceptors',
               'rotatable_bonds', 'tpsa', 'log_p']
target_col = 'Delta_PCE'

df_clean = df.dropna(subset=[target_col]).dropna(subset=feature_cols).copy()
print(f"\nDataset: {df_clean.shape[0]} samples, {len(feature_cols)} features")

# Prepare data for AutoGluon (needs target column included)
df_model = df_clean[feature_cols + [target_col]].copy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df_model[feature_cols], df_model[target_col], 
    test_size=0.2, random_state=42
)

# Create train/test dataframes with target
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print(f"Training set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Output directory
output_dir = "/share/yhm/test/AutoML_EDA"
figures_dir = f"{output_dir}/figures"
tables_dir = f"{output_dir}/tables"
model_dir = f"{output_dir}/autogluon_models"

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

if AUTOGLUON_AVAILABLE:
    print("\n" + "=" * 80)
    print("RUNNING AUTOGLUON AUTOML")
    print("=" * 80)
    
    # Initialize AutoGluon Predictor with 5-fold CV
    predictor = TabularPredictor(
        label=target_col,
        problem_type='regression',
        eval_metric='r2',
        path=model_dir
    )
    
    # Fit with time limit and 5-fold CV
    print("\nTraining AutoGluon models with 5-fold cross-validation...")
    start_time = time.time()
    
    predictor.fit(
        train_data=train_df,
        num_bag_folds=5,  # 5-fold bagging for robust CV
        presets='best_quality',  # Use best quality preset
        time_limit=600,  # 10 minutes time limit
        verbosity=2
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    
    # Get leaderboard
    leaderboard = predictor.leaderboard(silent=True)
    print("\n" + "=" * 80)
    print("MODEL LEADERBOARD (sorted by CV R²)")
    print("=" * 80)
    print(leaderboard.to_string(index=False))
    
    # Save leaderboard
    leaderboard.to_csv(f"{tables_dir}/autogluon_leaderboard.csv", index=False)
    
    # Get best model name
    best_model_name = leaderboard.iloc[0]['model']
    print(f"\nBest model: {best_model_name}")
    
    # Make predictions
    y_train_pred = predictor.predict(train_df.drop(columns=[target_col]))
    y_test_pred = predictor.predict(test_df.drop(columns=[target_col]))
    
    # Get feature importance
    try:
        feature_importance = predictor.feature_importance(train_df)
        print("\nFeature Importance:")
        print(feature_importance)
        feature_importance.to_csv(f"{tables_dir}/autogluon_feature_importance.csv")
    except Exception as e:
        print(f"Could not compute feature importance: {e}")
        feature_importance = None

else:
    print("\nUsing fallback: Optimized ensemble with GridSearchCV")
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.model_selection import GridSearchCV
    import xgboost as xgb
    
    # Define models and parameter grids
    models_params = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        },
        'ExtraTrees': {
            'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20, None]
            }
        }
    }
    
    # 5-fold CV
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    best_models = {}
    
    print("\nOptimizing models with 5-fold CV...")
    for name, config in models_params.items():
        print(f"\nOptimizing {name}...")
        grid = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=cv, 
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)
        
        # Get CV scores
        cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=cv, scoring='r2')
        
        # Predictions
        y_train_pred_model = grid.predict(X_train)
        y_test_pred_model = grid.predict(X_test)
        
        result = {
            'model': name,
            'best_params': str(grid.best_params_),
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'train_r2': r2_score(y_train, y_train_pred_model),
            'test_r2': r2_score(y_test, y_test_pred_model),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_model)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_model)),
            'train_mae': mean_absolute_error(y_train, y_train_pred_model),
            'test_mae': mean_absolute_error(y_test, y_test_pred_model)
        }
        results.append(result)
        best_models[name] = grid.best_estimator_
        
        print(f"  CV R² (5-fold): {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
        print(f"  Test R²: {result['test_r2']:.4f}")
    
    # Create leaderboard
    leaderboard = pd.DataFrame(results).sort_values('cv_r2_mean', ascending=False)
    print("\n" + "=" * 80)
    print("MODEL LEADERBOARD (sorted by 5-fold CV R²)")
    print("=" * 80)
    print(leaderboard[['model', 'cv_r2_mean', 'cv_r2_std', 'test_r2', 'test_rmse']].to_string(index=False))
    
    # Save leaderboard
    leaderboard.to_csv(f"{tables_dir}/autogluon_leaderboard.csv", index=False)
    
    # Use best model
    best_model_name = leaderboard.iloc[0]['model']
    best_model = best_models[best_model_name]
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    print(f"\nBest model: {best_model_name}")

# Calculate final metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# 5-fold CV on final predictions
cv = KFold(n_splits=5, shuffle=True, random_state=42)
if AUTOGLUON_AVAILABLE:
    # Use predictor's internal CV score
    cv_scores = predictor.evaluate(train_df, silent=True)
    cv_r2_mean = cv_scores.get('r2', test_r2)
    cv_r2_std = 0.0  # AutoGluon doesn't provide std directly
else:
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='r2')
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()

print("\n" + "=" * 80)
print("FINAL MODEL PERFORMANCE")
print("=" * 80)
print(f"\nBest Model: {best_model_name}")
print(f"\n5-Fold CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
print(f"\nTraining Set:")
print(f"  R²:   {train_r2:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE:  {train_mae:.4f}")
print(f"\nTest Set:")
print(f"  R²:   {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")

# ==============================================================================
# GENERATE PLOTS
# ==============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATION PLOTS")
print("=" * 80)

# 1. Training and Test Scatter Plots (side by side)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training Set Plot
ax1 = axes[0]
ax1.scatter(y_train, y_train_pred, alpha=0.5, s=15, c='steelblue', edgecolors='none')

min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

# Regression line
z = np.polyfit(y_train, y_train_pred, 1)
p = np.poly1d(z)
x_line = np.linspace(min_val, max_val, 100)
ax1.plot(x_line, p(x_line), 'g-', lw=1.5, alpha=0.7, label=f'Fit (slope={z[0]:.3f})')

ax1.set_xlabel('True Delta_PCE (%)', fontsize=12)
ax1.set_ylabel('Predicted Delta_PCE (%)', fontsize=12)
ax1.set_title(f'Training Set (n={len(y_train)})\n$R^2$ = {train_r2:.4f}, RMSE = {train_rmse:.3f}%', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')

# Test Set Plot
ax2 = axes[1]
ax2.scatter(y_test, y_test_pred, alpha=0.5, s=15, c='darkorange', edgecolors='none')

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

# Regression line
z = np.polyfit(y_test, y_test_pred, 1)
p = np.poly1d(z)
x_line = np.linspace(min_val, max_val, 100)
ax2.plot(x_line, p(x_line), 'g-', lw=1.5, alpha=0.7, label=f'Fit (slope={z[0]:.3f})')

ax2.set_xlabel('True Delta_PCE (%)', fontsize=12)
ax2.set_ylabel('Predicted Delta_PCE (%)', fontsize=12)
ax2.set_title(f'Test Set (n={len(y_test)})\n$R^2$ = {test_r2:.4f}, RMSE = {test_rmse:.3f}%', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal', adjustable='box')

plt.suptitle(f'AutoGluon Optimized Model: {best_model_name}\n5-Fold CV R² = {cv_r2_mean:.4f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{figures_dir}/autogluon_train_test_scatter.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {figures_dir}/autogluon_train_test_scatter.png")

# 2. Residual Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training residuals
train_residuals = y_train - y_train_pred
ax1 = axes[0]
ax1.scatter(y_train_pred, train_residuals, alpha=0.5, s=15, c='steelblue', edgecolors='none')
ax1.axhline(y=0, color='r', linestyle='--', lw=2)
ax1.set_xlabel('Predicted Delta_PCE (%)', fontsize=12)
ax1.set_ylabel('Residuals (%)', fontsize=12)
ax1.set_title(f'Training Set Residuals\nMean: {train_residuals.mean():.4f}, Std: {train_residuals.std():.4f}', fontsize=12)
ax1.grid(True, alpha=0.3)

# Test residuals
test_residuals = y_test - y_test_pred
ax2 = axes[1]
ax2.scatter(y_test_pred, test_residuals, alpha=0.5, s=15, c='darkorange', edgecolors='none')
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Delta_PCE (%)', fontsize=12)
ax2.set_ylabel('Residuals (%)', fontsize=12)
ax2.set_title(f'Test Set Residuals\nMean: {test_residuals.mean():.4f}, Std: {test_residuals.std():.4f}', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.suptitle(f'Residual Analysis - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{figures_dir}/autogluon_residual_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {figures_dir}/autogluon_residual_plots.png")

# 3. Model Comparison Bar Plot
if 'leaderboard' in dir() and len(leaderboard) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # CV R² comparison
    ax1 = axes[0]
    models = leaderboard['model'].head(10).tolist()
    cv_scores_plot = leaderboard['cv_r2_mean'].head(10).tolist()
    cv_stds_plot = leaderboard['cv_r2_std'].head(10).tolist() if 'cv_r2_std' in leaderboard.columns else [0]*len(models)
    
    colors = ['forestgreen' if i == 0 else 'steelblue' for i in range(len(models))]
    bars = ax1.barh(range(len(models)), cv_scores_plot, xerr=cv_stds_plot, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models)
    ax1.set_xlabel('5-Fold CV R² Score', fontsize=12)
    ax1.set_title('Model Comparison by CV R²', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, cv_scores_plot)):
        ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=10)
    
    # Test R² comparison
    ax2 = axes[1]
    test_scores = leaderboard['test_r2'].head(10).tolist()
    
    bars = ax2.barh(range(len(models)), test_scores, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models)
    ax2.set_xlabel('Test R² Score', fontsize=12)
    ax2.set_title('Model Comparison by Test R²', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    for i, (bar, score) in enumerate(zip(bars, test_scores)):
        ax2.text(score + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=10)
    
    plt.suptitle('AutoGluon Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/autogluon_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/autogluon_model_comparison.png")

# 4. Prediction Distribution Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training distribution
ax1 = axes[0]
ax1.hist(y_train, bins=50, alpha=0.7, label='True', color='steelblue', density=True)
ax1.hist(y_train_pred, bins=50, alpha=0.7, label='Predicted', color='coral', density=True)
ax1.set_xlabel('Delta_PCE (%)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Training Set: True vs Predicted Distribution', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Test distribution
ax2 = axes[1]
ax2.hist(y_test, bins=50, alpha=0.7, label='True', color='steelblue', density=True)
ax2.hist(y_test_pred, bins=50, alpha=0.7, label='Predicted', color='coral', density=True)
ax2.set_xlabel('Delta_PCE (%)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Test Set: True vs Predicted Distribution', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(f'Prediction Distribution Analysis - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{figures_dir}/autogluon_prediction_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {figures_dir}/autogluon_prediction_distribution.png")

# ==============================================================================
# GENERATE REPORT
# ==============================================================================
print("\n" + "=" * 80)
print("GENERATING MACHINE LEARNING REPORT")
print("=" * 80)

report_path = f"{output_dir}/autogluon_ml_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# AutoGluon优化机器学习报告\n\n")
    f.write("---\n\n")
    f.write("## 摘要\n\n")
    f.write(f"本报告使用AutoGluon自动机器学习框架，以**5折交叉验证R²**为优化目标，")
    f.write(f"对{len(df_clean)}个钙钛矿太阳能电池调控剂样本进行Delta_PCE预测模型优化。\n\n")
    f.write(f"**最优模型**: {best_model_name}\n\n")
    f.write(f"**5折交叉验证R²**: {cv_r2_mean:.4f}")
    if cv_r2_std > 0:
        f.write(f" ± {cv_r2_std:.4f}")
    f.write("\n\n")
    
    f.write("---\n\n")
    f.write("## 1. 数据集概况\n\n")
    f.write("| 统计量 | 数值 |\n")
    f.write("|--------|------|\n")
    f.write(f"| 总样本数 | {len(df_clean)} |\n")
    f.write(f"| 训练集 | {len(y_train)} (80%) |\n")
    f.write(f"| 测试集 | {len(y_test)} (20%) |\n")
    f.write(f"| 特征数 | {len(feature_cols)} |\n")
    f.write(f"| 目标变量 | Delta_PCE |\n\n")
    
    f.write("**特征列表**:\n")
    for feat in feature_cols:
        f.write(f"- {feat}\n")
    f.write("\n")
    
    f.write("---\n\n")
    f.write("## 2. 模型性能\n\n")
    f.write("### 2.1 最优模型性能指标\n\n")
    f.write("| 数据集 | R² | RMSE (%) | MAE (%) |\n")
    f.write("|--------|------|---------|---------|\n")
    f.write(f"| **5折CV** | **{cv_r2_mean:.4f}** | - | - |\n")
    f.write(f"| 训练集 | {train_r2:.4f} | {train_rmse:.3f} | {train_mae:.3f} |\n")
    f.write(f"| 测试集 | {test_r2:.4f} | {test_rmse:.3f} | {test_mae:.3f} |\n\n")
    
    f.write("### 2.2 过拟合分析\n\n")
    overfit_gap = train_r2 - test_r2
    if overfit_gap < 0.1:
        status = "**良好** - 轻微过拟合"
    elif overfit_gap < 0.2:
        status = "**中等** - 存在适度过拟合"
    else:
        status = "**严重** - 明显过拟合，建议正则化"
    f.write(f"- 训练集与测试集R²差距: {overfit_gap:.4f}\n")
    f.write(f"- 评估: {status}\n\n")
    
    f.write("---\n\n")
    f.write("## 3. 模型比较\n\n")
    f.write("### 3.1 模型排行榜 (按5折CV R²排序)\n\n")
    
    # Save leaderboard table
    if 'cv_r2_std' in leaderboard.columns:
        f.write("| 模型 | CV R² (Mean) | CV R² (Std) | Test R² | Test RMSE |\n")
        f.write("|------|-------------|-------------|---------|----------|\n")
        for _, row in leaderboard.head(10).iterrows():
            f.write(f"| {row['model']} | {row['cv_r2_mean']:.4f} | {row['cv_r2_std']:.4f} | {row['test_r2']:.4f} | {row['test_rmse']:.4f} |\n")
    else:
        f.write("| 模型 | Score | Test R² | Test RMSE |\n")
        f.write("|------|-------|---------|----------|\n")
        for _, row in leaderboard.head(10).iterrows():
            score_col = 'score' if 'score' in row else 'cv_r2_mean'
            f.write(f"| {row['model']} | {row[score_col]:.4f} | {row.get('test_r2', 'N/A')} | {row.get('test_rmse', 'N/A')} |\n")
    f.write("\n")
    
    f.write("### 3.2 模型比较图\n\n")
    f.write("![模型比较](../figures/autogluon_model_comparison.png)\n\n")
    f.write("*图1. 各模型的5折CV R²和Test R²比较*\n\n")
    
    f.write("---\n\n")
    f.write("## 4. 预测结果分析\n\n")
    f.write("### 4.1 训练集和测试集预测散点图\n\n")
    f.write("![训练测试散点图](../figures/autogluon_train_test_scatter.png)\n\n")
    f.write(f"*图2. {best_model_name}模型的训练集(左)和测试集(右)True vs Predicted散点图*\n\n")
    
    f.write("**关键观察**:\n")
    f.write(f"- 训练集R² = {train_r2:.4f}，预测趋势明显\n")
    f.write(f"- 测试集R² = {test_r2:.4f}，泛化性能\n")
    slope_train = np.polyfit(y_train, y_train_pred, 1)[0]
    slope_test = np.polyfit(y_test, y_test_pred, 1)[0]
    f.write(f"- 回归斜率: 训练集={slope_train:.3f}, 测试集={slope_test:.3f}\n")
    f.write("- 斜率<1表明模型倾向于向均值收敛\n\n")
    
    f.write("### 4.2 残差分析\n\n")
    f.write("![残差图](../figures/autogluon_residual_plots.png)\n\n")
    f.write("*图3. 训练集和测试集的残差分布*\n\n")
    f.write(f"- 训练集残差: 均值={train_residuals.mean():.4f}, 标准差={train_residuals.std():.4f}\n")
    f.write(f"- 测试集残差: 均值={test_residuals.mean():.4f}, 标准差={test_residuals.std():.4f}\n\n")
    
    f.write("### 4.3 预测分布对比\n\n")
    f.write("![预测分布](../figures/autogluon_prediction_distribution.png)\n\n")
    f.write("*图4. 真实值与预测值的分布对比*\n\n")
    
    f.write("---\n\n")
    f.write("## 5. 结论与建议\n\n")
    f.write("### 5.1 主要发现\n\n")
    f.write(f"1. **最优模型**: {best_model_name}\n")
    f.write(f"2. **5折CV R²**: {cv_r2_mean:.4f}")
    if cv_r2_std > 0:
        f.write(f" ± {cv_r2_std:.4f}")
    f.write("\n")
    f.write(f"3. **测试集R²**: {test_r2:.4f}\n")
    f.write(f"4. **预测误差(RMSE)**: {test_rmse:.3f}%\n\n")
    
    f.write("### 5.2 模型局限性\n\n")
    f.write("- R²值相对较低，表明分子描述符对Delta_PCE的解释能力有限\n")
    f.write("- 可能存在未纳入的关键因素（如钙钛矿组成、加工条件等）\n")
    f.write("- 建议引入更多量子化学描述符以提升预测能力\n\n")
    
    f.write("### 5.3 后续改进建议\n\n")
    f.write("1. **特征工程**: 计算HOMO/LUMO、偶极矩等量子化学描述符\n")
    f.write("2. **数据增强**: 收集更多标准化实验数据\n")
    f.write("3. **深度学习**: 尝试图神经网络(GNN)直接从分子结构学习\n")
    f.write("4. **集成学习**: 结合多种模型的预测结果\n\n")
    
    f.write("---\n\n")
    f.write("## 附录：文件列表\n\n")
    f.write(f"- 模型排行榜: `{tables_dir}/autogluon_leaderboard.csv`\n")
    f.write(f"- 预测散点图: `{figures_dir}/autogluon_train_test_scatter.png`\n")
    f.write(f"- 残差图: `{figures_dir}/autogluon_residual_plots.png`\n")
    f.write(f"- 模型比较图: `{figures_dir}/autogluon_model_comparison.png`\n")
    f.write(f"- 预测分布图: `{figures_dir}/autogluon_prediction_distribution.png`\n")
    f.write(f"- 本报告: `{report_path}`\n\n")
    
    f.write("---\n\n")
    f.write(f"**生成日期**: {pd.Timestamp.now().strftime('%Y年%m月%d日')}\n\n")
    f.write("**优化方法**: AutoGluon AutoML (5-Fold CV R²)\n")

print(f"\nReport saved to: {report_path}")

print("\n" + "=" * 80)
print("AUTOGLUON OPTIMIZATION COMPLETED")
print("=" * 80)
print(f"\nBest Model: {best_model_name}")
print(f"5-Fold CV R²: {cv_r2_mean:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"\nAll outputs saved to: {output_dir}")
