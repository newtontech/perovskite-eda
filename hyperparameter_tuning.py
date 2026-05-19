#!/usr/bin/env python3
"""
Fast Hyperparameter Tuning for QSPR Machine Learning Models
Target: Delta_PCE prediction using molecular descriptors
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HYPERPARAMETER TUNING FOR QSPR DELTA_PCE PREDICTION")
print("=" * 80)

# Load data
data_path = "/share/yhm/test/AutoML_EDA/processed_data.csv"
df = pd.read_csv(data_path)

feature_cols = ['molecular_weight', 'h_bond_donors', 'h_bond_acceptors',
               'rotatable_bonds', 'tpsa', 'log_p']
target_col = 'Delta_PCE'

df_clean = df.dropna(subset=[target_col]).dropna(subset=feature_cols)
X = df_clean[feature_cols].values
y = df_clean[target_col].values

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target: Mean={np.mean(y):.4f}, Std={np.std(y):.4f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grids (reduced for speed)
param_grids = {
    'Ridge': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None]},
    'XGBoost': {'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [3, 5, 7], 'n_estimators': [100, 200]},
    'SVR': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
}

# Models
models = {
    'Ridge': Ridge(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=1),
    'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=1),
    'SVR': SVR()
}

results = {}
best_models = {}

print("\n" + "=" * 80)
print("PERFORMING GRID SEARCH")
print("=" * 80)

for name, model in models.items():
    print(f"\n{name}...")
    grid = GridSearchCV(model, param_grids[name], cv=3, scoring='r2', n_jobs=1)
    grid.fit(X_train_scaled, y_train)

    best_models[name] = grid.best_estimator_

    y_pred_test = grid.predict(X_test_scaled)
    y_pred_train = grid.predict(X_train_scaled)

    results[name] = {
        'best_params': grid.best_params_,
        'best_cv_score': grid.best_score_,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }

    print(f"  Best params: {grid.best_params_}")
    print(f"  Test R²: {results[name]['test_r2']:.4f}")

# Save results
output_dir = "/share/yhm/test/AutoML_EDA"

# Save best hyperparameters
best_params_data = []
for model_name, model_results in results.items():
    params = model_results['best_params'].copy()
    params['model'] = model_name
    params['best_cv_r2'] = model_results['best_cv_score']
    best_params_data.append(params)

best_params_df = pd.DataFrame(best_params_data)
best_params_df.to_csv(f"{output_dir}/tables/best_hyperparameters.csv", index=False)

# Save performance comparison
performance_data = []
for model_name, model_results in results.items():
    performance_data.append({
        'Model': model_name,
        'CV_R2': model_results['best_cv_score'],
        'Train_R2': model_results['train_r2'],
        'Test_R2': model_results['test_r2'],
        'Test_RMSE': model_results['test_rmse'],
        'Test_MAE': model_results['test_mae']
    })

performance_df = pd.DataFrame(performance_data)
performance_df = performance_df.sort_values('Test_R2', ascending=False)
performance_df.to_csv(f"{output_dir}/tables/model_performance_comparison.csv", index=False)

# Create learning curves
print("\n" + "=" * 80)
print("CREATING LEARNING CURVES")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()
train_sizes = np.linspace(0.1, 1.0, 10)

for idx, (name, model) in enumerate(best_models.items()):
    print(f"Learning curve for {name}...")
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train, train_sizes=train_sizes, cv=3, scoring='r2', n_jobs=1
    )

    ax = axes[idx]
    ax.plot(train_sizes_abs, np.mean(train_scores, axis=1), 'o-', label='Training', color='blue')
    ax.plot(train_sizes_abs, np.mean(val_scores, axis=1), 's-', label='Validation', color='red')
    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel('R² Score')
    ax.set_title(f'{name} Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/figures/learning_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# Create validation curves
print("\n" + "=" * 80)
print("CREATING VALIDATION CURVES")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

# Ridge alpha validation curve
print("Ridge validation curve...")
param_range = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores, val_scores = validation_curve(
    Ridge(random_state=42), X_train_scaled, y_train,
    param_name='alpha', param_range=param_range, cv=3, scoring='r2', n_jobs=1
)
ax = axes[0]
ax.plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training', color='blue')
ax.plot(param_range, np.mean(val_scores, axis=1), 's-', label='Validation', color='red')
ax.set_xscale('log')
ax.set_xlabel('Alpha')
ax.set_ylabel('R² Score')
ax.set_title('Ridge Validation Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# RF max_depth validation curve
print("Random Forest validation curve...")
param_range = [5, 10, 15, 20, 25, None]
train_scores, val_scores = validation_curve(
    RandomForestRegressor(random_state=42, n_jobs=1), X_train_scaled, y_train,
    param_name='max_depth', param_range=param_range, cv=3, scoring='r2', n_jobs=1
)
ax = axes[1]
ax.plot(range(len(param_range)), np.mean(train_scores, axis=1), 'o-', label='Training', color='blue')
ax.plot(range(len(param_range)), np.mean(val_scores, axis=1), 's-', label='Validation', color='red')
ax.set_xticks(range(len(param_range)))
ax.set_xticklabels([str(p) for p in param_range])
ax.set_xlabel('Max Depth')
ax.set_ylabel('R² Score')
ax.set_title('Random Forest Validation Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# XGBoost learning_rate validation curve
print("XGBoost validation curve...")
param_range = [0.01, 0.05, 0.1, 0.2, 0.3]
train_scores, val_scores = validation_curve(
    xgb.XGBRegressor(random_state=42, n_jobs=1), X_train_scaled, y_train,
    param_name='learning_rate', param_range=param_range, cv=3, scoring='r2', n_jobs=1
)
ax = axes[2]
ax.plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training', color='blue')
ax.plot(param_range, np.mean(val_scores, axis=1), 's-', label='Validation', color='red')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('R² Score')
ax.set_title('XGBoost Validation Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# SVR C validation curve
print("SVR validation curve...")
param_range = [0.1, 1, 10, 50, 100]
train_scores, val_scores = validation_curve(
    SVR(kernel='rbf'), X_train_scaled, y_train,
    param_name='C', param_range=param_range, cv=3, scoring='r2', n_jobs=1
)
ax = axes[3]
ax.plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training', color='blue')
ax.plot(param_range, np.mean(val_scores, axis=1), 's-', label='Validation', color='red')
ax.set_xscale('log')
ax.set_xlabel('C')
ax.set_ylabel('R² Score')
ax.set_title('SVR Validation Curve')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/figures/validation_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# Create prediction vs actual plots
print("\n" + "=" * 80)
print("CREATING PREDICTION PLOTS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, (name, model) in enumerate(best_models.items()):
    y_pred = model.predict(X_test_scaled)

    ax = axes[idx]
    ax.scatter(y_test, y_pred, alpha=0.5, s=20)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    r2 = results[name]['test_r2']
    rmse = results[name]['test_rmse']

    ax.set_xlabel('Actual Delta_PCE')
    ax.set_ylabel('Predicted Delta_PCE')
    ax.set_title(f'{name}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/figures/prediction_vs_actual.png", dpi=300, bbox_inches='tight')
plt.close()

# Create residual plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, (name, model) in enumerate(best_models.items()):
    y_pred = model.predict(X_test_scaled)
    residuals = y_test - y_pred

    ax = axes[idx]
    ax.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)

    ax.set_xlabel('Predicted Delta_PCE')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{name} Residual Plot')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/figures/residual_plots.png", dpi=300, bbox_inches='tight')
plt.close()

# Save report
report_path = f"{output_dir}/hyperparameter_tuning_report.txt"
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("HYPERPARAMETER TUNING REPORT FOR QSPR DELTA_PCE PREDICTION\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATASET:\n")
    f.write(f"  Samples: {X.shape[0]}\n")
    f.write(f"  Features: {X.shape[1]}\n")
    f.write(f"  Train/Test: {X_train.shape[0]}/{X_test.shape[0]}\n\n")

    f.write("BEST HYPERPARAMETERS:\n")
    f.write("-" * 80 + "\n")
    for model_name, model_results in results.items():
        f.write(f"\n{model_name}:\n")
        for param, value in model_results['best_params'].items():
            f.write(f"  {param}: {value}\n")

    f.write("\n\nMODEL PERFORMANCE:\n")
    f.write("-" * 80 + "\n")
    for _, row in performance_df.iterrows():
        f.write(f"\n{row['Model']}:\n")
        f.write(f"  CV R²:      {row['CV_R2']:.4f}\n")
        f.write(f"  Train R²:   {row['Train_R2']:.4f}\n")
        f.write(f"  Test R²:    {row['Test_R2']:.4f}\n")
        f.write(f"  Test RMSE:  {row['Test_RMSE']:.4f}\n")
        f.write(f"  Test MAE:   {row['Test_MAE']:.4f}\n")

    f.write("\n\nKEY FINDINGS:\n")
    f.write("-" * 80 + "\n")
    best_model = performance_df.iloc[0]
    f.write(f"Best model: {best_model['Model']}\n")
    f.write(f"Best Test R²: {best_model['Test_R2']:.4f}\n")
    f.write(f"Best Test RMSE: {best_model['Test_RMSE']:.4f}\n\n")

    f.write("Overfitting analysis:\n")
    for model_name, model_results in results.items():
        diff = model_results['train_r2'] - model_results['test_r2']
        status = "Good" if diff < 0.1 else "Moderate" if diff < 0.2 else "Severe"
        f.write(f"  {model_name}: gap = {diff:.4f} ({status} overfitting)\n")

    f.write("\n\nRECOMMENDATIONS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"1. Use {best_model['Model']} for production\n")
    f.write(f"2. Expected prediction error (RMSE): {best_model['Test_RMSE']:.4f}\n")
    f.write("3. Consider ensemble methods for improved performance\n")
    f.write("4. Explore feature engineering with chemical domain knowledge\n")

print("\n" + "=" * 80)
print("COMPLETED")
print("=" * 80)
print("\nFiles created:")
print(f"  - {output_dir}/tables/best_hyperparameters.csv")
print(f"  - {output_dir}/tables/model_performance_comparison.csv")
print(f"  - {output_dir}/figures/learning_curves.png")
print(f"  - {output_dir}/figures/validation_curves.png")
print(f"  - {output_dir}/figures/prediction_vs_actual.png")
print(f"  - {output_dir}/figures/residual_plots.png")
print(f"  - {output_dir}/hyperparameter_tuning_report.txt")

print("\n" + "=" * 80)
print("MODEL RANKING (by Test R²)")
print("=" * 80)
print(performance_df.to_string(index=False))
