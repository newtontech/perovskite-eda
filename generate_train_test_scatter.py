#!/usr/bin/env python3
"""
Generate Training and Test Set True vs Predicted PCE Scatter Plots
for QSPR Analysis Report
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GENERATING TRAIN/TEST SCATTER PLOTS FOR QSPR REPORT")
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

# Split data with same random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest (best model from previous analysis)
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Get predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nTraining Set: R²={train_r2:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
print(f"Test Set: R²={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

# Create figure with training and test scatter plots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training Set Plot
ax1 = axes[0]
ax1.scatter(y_train, y_train_pred, alpha=0.5, s=15, c='steelblue', edgecolors='none')

# Perfect prediction line
min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

# Add regression line
z = np.polyfit(y_train, y_train_pred, 1)
p = np.poly1d(z)
x_line = np.linspace(min_val, max_val, 100)
ax1.plot(x_line, p(x_line), 'g-', lw=1.5, alpha=0.7, label=f'Regression (slope={z[0]:.3f})')

ax1.set_xlabel('True Delta_PCE (%)', fontsize=12)
ax1.set_ylabel('Predicted Delta_PCE (%)', fontsize=12)
ax1.set_title(f'Training Set (n={len(y_train)})\n$R^2$ = {train_r2:.4f}, RMSE = {train_rmse:.3f}%, MAE = {train_mae:.3f}%', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')

# Test Set Plot
ax2 = axes[1]
ax2.scatter(y_test, y_test_pred, alpha=0.5, s=15, c='darkorange', edgecolors='none')

# Perfect prediction line
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

# Add regression line
z = np.polyfit(y_test, y_test_pred, 1)
p = np.poly1d(z)
x_line = np.linspace(min_val, max_val, 100)
ax2.plot(x_line, p(x_line), 'g-', lw=1.5, alpha=0.7, label=f'Regression (slope={z[0]:.3f})')

ax2.set_xlabel('True Delta_PCE (%)', fontsize=12)
ax2.set_ylabel('Predicted Delta_PCE (%)', fontsize=12)
ax2.set_title(f'Test Set (n={len(y_test)})\n$R^2$ = {test_r2:.4f}, RMSE = {test_rmse:.3f}%, MAE = {test_mae:.3f}%', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal', adjustable='box')

plt.tight_layout()
output_path = "/share/yhm/test/AutoML_EDA/figures/train_test_scatter_plots.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nScatter plot saved to: {output_path}")

# Create a combined figure with all models comparison
print("\nGenerating comparison plot for multiple models...")

models = {
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

fig, axes = plt.subplots(3, 2, figsize=(14, 18))

for row_idx, (name, model) in enumerate(models.items()):
    print(f"Training {name}...")
    
    # Use scaled data for Ridge, original for tree-based
    if name == 'Ridge Regression':
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Training Set Plot
    ax1 = axes[row_idx, 0]
    ax1.scatter(y_train, y_train_pred, alpha=0.4, s=10, c='steelblue', edgecolors='none')
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('True Delta_PCE (%)', fontsize=11)
    ax1.set_ylabel('Predicted Delta_PCE (%)', fontsize=11)
    ax1.set_title(f'{name} - Training Set\n$R^2$ = {train_r2:.4f}, RMSE = {train_rmse:.3f}%', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Test Set Plot
    ax2 = axes[row_idx, 1]
    ax2.scatter(y_test, y_test_pred, alpha=0.4, s=10, c='darkorange', edgecolors='none')
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel('True Delta_PCE (%)', fontsize=11)
    ax2.set_ylabel('Predicted Delta_PCE (%)', fontsize=11)
    ax2.set_title(f'{name} - Test Set\n$R^2$ = {test_r2:.4f}, RMSE = {test_rmse:.3f}%', fontsize=11)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path2 = "/share/yhm/test/AutoML_EDA/figures/all_models_train_test_scatter.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Multi-model comparison saved to: {output_path2}")

print("\n" + "=" * 80)
print("COMPLETED")
print("=" * 80)
