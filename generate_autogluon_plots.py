#!/usr/bin/env python3
"""
Generate plots for AutoGluon results
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

print(f"Best Model: {best_model_name}")
print(f"5-Fold CV R²: {cv_r2:.4f}")
print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

output_dir = "/share/yhm/test/AutoML_EDA"
figures_dir = f"{output_dir}/figures"

# 1. Training and Test Scatter Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training Set Plot
ax1 = axes[0]
ax1.scatter(y_train, y_train_pred, alpha=0.5, s=15, c='steelblue', edgecolors='none')

min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

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

plt.suptitle(f'AutoGluon Optimized Model: {best_model_name}\n5-Fold CV $R^2$ = {cv_r2:.4f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{figures_dir}/autogluon_train_test_scatter.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {figures_dir}/autogluon_train_test_scatter.png")

# 2. Residual Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

train_residuals = y_train - y_train_pred
ax1 = axes[0]
ax1.scatter(y_train_pred, train_residuals, alpha=0.5, s=15, c='steelblue', edgecolors='none')
ax1.axhline(y=0, color='r', linestyle='--', lw=2)
ax1.set_xlabel('Predicted Delta_PCE (%)', fontsize=12)
ax1.set_ylabel('Residuals (%)', fontsize=12)
ax1.set_title(f'Training Set Residuals\nMean: {train_residuals.mean():.4f}, Std: {train_residuals.std():.4f}', fontsize=12)
ax1.grid(True, alpha=0.3)

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
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# CV R² comparison
ax1 = axes[0]
models = leaderboard['model'].head(10).tolist()
scores = leaderboard['score_val'].head(10).tolist()

colors = ['forestgreen' if i == 0 else 'steelblue' for i in range(len(models))]
bars = ax1.barh(range(len(models)), scores, color=colors, alpha=0.8)
ax1.set_yticks(range(len(models)))
ax1.set_yticklabels([m[:25] for m in models], fontsize=9)
ax1.set_xlabel('5-Fold CV R² Score', fontsize=12)
ax1.set_title('Model Comparison by CV R²', fontsize=12)
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()
ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

for i, (bar, score) in enumerate(zip(bars, scores)):
    ax1.text(score + 0.005 if score >= 0 else score - 0.02, 
             bar.get_y() + bar.get_height()/2, 
             f'{score:.4f}', va='center', fontsize=9)

# Feature Importance
ax2 = axes[1]
try:
    feature_importance = predictor.feature_importance(train_df)
    features = feature_importance.index.tolist()
    importances = feature_importance['importance'].tolist()
    
    colors = ['coral' if imp < 0 else 'steelblue' for imp in importances]
    bars = ax2.barh(range(len(features)), importances, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels(features, fontsize=10)
    ax2.set_xlabel('Feature Importance', fontsize=12)
    ax2.set_title('Feature Importance', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    for bar, imp in zip(bars, importances):
        ax2.text(imp + 0.005 if imp >= 0 else imp - 0.02, 
                 bar.get_y() + bar.get_height()/2, 
                 f'{imp:.4f}', va='center', fontsize=9)
except:
    ax2.text(0.5, 0.5, 'Feature importance\nnot available', 
             ha='center', va='center', transform=ax2.transAxes, fontsize=14)

plt.suptitle('AutoGluon Model Performance & Feature Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{figures_dir}/autogluon_model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {figures_dir}/autogluon_model_comparison.png")

# 4. Prediction Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
ax1.hist(y_train, bins=50, alpha=0.7, label='True', color='steelblue', density=True)
ax1.hist(y_train_pred, bins=50, alpha=0.7, label='Predicted', color='coral', density=True)
ax1.set_xlabel('Delta_PCE (%)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Training Set: True vs Predicted Distribution', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

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

# 5. All models comparison scatter (top 3 models)
fig, axes = plt.subplots(3, 2, figsize=(14, 18))

top_models = leaderboard['model'].head(3).tolist()

for row_idx, model_name in enumerate(top_models):
    # Get predictions for this model
    y_train_pred_m = predictor.predict(train_df.drop(columns=[target_col]), model=model_name)
    y_test_pred_m = predictor.predict(test_df.drop(columns=[target_col]), model=model_name)
    
    train_r2_m = r2_score(y_train, y_train_pred_m)
    test_r2_m = r2_score(y_test, y_test_pred_m)
    train_rmse_m = np.sqrt(mean_squared_error(y_train, y_train_pred_m))
    test_rmse_m = np.sqrt(mean_squared_error(y_test, y_test_pred_m))
    
    # Training Set Plot
    ax1 = axes[row_idx, 0]
    ax1.scatter(y_train, y_train_pred_m, alpha=0.4, s=10, c='steelblue', edgecolors='none')
    min_val = min(y_train.min(), y_train_pred_m.min())
    max_val = max(y_train.max(), y_train_pred_m.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('True Delta_PCE (%)', fontsize=11)
    ax1.set_ylabel('Predicted Delta_PCE (%)', fontsize=11)
    ax1.set_title(f'{model_name[:30]} - Training\n$R^2$ = {train_r2_m:.4f}, RMSE = {train_rmse_m:.3f}%', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Test Set Plot
    ax2 = axes[row_idx, 1]
    ax2.scatter(y_test, y_test_pred_m, alpha=0.4, s=10, c='darkorange', edgecolors='none')
    min_val = min(y_test.min(), y_test_pred_m.min())
    max_val = max(y_test.max(), y_test_pred_m.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel('True Delta_PCE (%)', fontsize=11)
    ax2.set_ylabel('Predicted Delta_PCE (%)', fontsize=11)
    ax2.set_title(f'{model_name[:30]} - Test\n$R^2$ = {test_r2_m:.4f}, RMSE = {test_rmse_m:.3f}%', fontsize=11)
    ax2.grid(True, alpha=0.3)

plt.suptitle('Top 3 AutoGluon Models: Training vs Test Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{figures_dir}/autogluon_top3_models_scatter.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {figures_dir}/autogluon_top3_models_scatter.png")

print("\nAll plots generated successfully!")
