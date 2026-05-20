#!/usr/bin/env python3
"""
QSPR ML Model Building Script for Delta_PCE Prediction
========================================================

This script performs the following:
1. Load and preprocess data from the Excel file
2. Calculate Delta_PCE (jv_reverse_scan_pce - jv_reverse_scan_pce_without_modulator)
3. Train multiple ML models with cross-validation
4. Evaluate models using R², RMSE, MAE
5. Generate feature importance and SHAP analysis
6. Save all results and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP - handle if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. SHAP plots will be skipped.")

import os

# Configuration
DATA_PATH = '/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx'
OUTPUT_DIR = '/share/yhm/test/AutoML_EDA'
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, 'processed_data.csv')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Feature columns (chemical descriptors)
FEATURE_COLS = [
    'molecular_weight',
    'h_bond_donors',
    'h_bond_acceptors',
    'rotatable_bonds',
    'tpsa',
    'log_p'
]

# Target calculation columns
PCE_WITH_MOD = 'jv_reverse_scan_pce'
PCE_WITHOUT_MOD = 'jv_reverse_scan_pce_without_modulator'
TARGET_COL = 'Delta_PCE'

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 60)
print("QSPR ML Model Building for Delta_PCE Prediction")
print("=" * 60)

# ============================================================================
# STEP 1: Load and Preprocess Data
# ============================================================================
print("\n[STEP 1] Loading and preprocessing data...")

# Load data
print(f"Loading data from: {DATA_PATH}")
df = pd.read_excel(DATA_PATH)
print(f"Original data shape: {df.shape}")

# Check column availability
print("\nChecking column availability:")
for col in FEATURE_COLS + [PCE_WITH_MOD, PCE_WITHOUT_MOD]:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null} non-null values ({100*non_null/len(df):.1f}%)")
    else:
        print(f"  {col}: NOT FOUND")

# Calculate Delta_PCE
print(f"\nCalculating {TARGET_COL}...")
df[TARGET_COL] = df[PCE_WITH_MOD] - df[PCE_WITHOUT_MOD]

# Filter for complete cases
print("\nFiltering for complete cases...")
feature_mask = df[FEATURE_COLS].notna().all(axis=1)
target_mask = df[TARGET_COL].notna()
complete_mask = feature_mask & target_mask

print(f"  Rows with all features: {feature_mask.sum()} ({100*feature_mask.sum()/len(df):.1f}%)")
print(f"  Rows with valid target: {target_mask.sum()} ({100*target_mask.sum()/len(df):.1f}%)")
print(f"  Complete cases: {complete_mask.sum()} ({100*complete_mask.sum()/len(df):.1f}%)")

# Create clean dataset
df_clean = df[complete_mask][FEATURE_COLS + [TARGET_COL]].copy()
print(f"\nClean dataset shape: {df_clean.shape}")

# Save processed data
df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"Processed data saved to: {PROCESSED_DATA_PATH}")

# Display statistics
print(f"\n{TARGET_COL} Statistics:")
print(f"  Mean: {df_clean[TARGET_COL].mean():.4f}")
print(f"  Std: {df_clean[TARGET_COL].std():.4f}")
print(f"  Min: {df_clean[TARGET_COL].min():.4f}")
print(f"  Max: {df_clean[TARGET_COL].max():.4f}")
print(f"  Median: {df_clean[TARGET_COL].median():.4f}")

# ============================================================================
# STEP 2: Prepare Data for ML
# ============================================================================
print("\n[STEP 2] Preparing data for machine learning...")

X = df_clean[FEATURE_COLS].values
y = df_clean[TARGET_COL].values

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 3: Define and Train Models
# ============================================================================
print("\n[STEP 3] Training and evaluating models...")

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

# Cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Store results
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Use scaled data for Linear, Ridge, and SVR; original data for tree-based
    if name in ['Random Forest', 'Gradient Boosting']:
        X_train_use = X_train
        X_test_use = X_test
    else:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled

    # Train model
    model.fit(X_train_use, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_use)
    y_test_pred = model.predict(X_test_use)

    # Cross-validation scores
    cv_r2 = cross_val_score(model, X_train_use, y_train, cv=cv, scoring='r2', n_jobs=-1)
    cv_rmse = -cross_val_score(model, X_train_use, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_mae = -cross_val_score(model, X_train_use, y_train, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)

    # Calculate metrics
    result = {
        'Model': name,
        'Train_R2': r2_score(y_train, y_train_pred),
        'Test_R2': r2_score(y_test, y_test_pred),
        'CV_R2_Mean': cv_r2.mean(),
        'CV_R2_Std': cv_r2.std(),
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'CV_RMSE_Mean': cv_rmse.mean(),
        'CV_RMSE_Std': cv_rmse.std(),
        'Train_MAE': mean_absolute_error(y_train, y_train_pred),
        'Test_MAE': mean_absolute_error(y_test, y_test_pred),
        'CV_MAE_Mean': cv_mae.mean(),
        'CV_MAE_Std': cv_mae.std()
    }
    results.append(result)

    print(f"  Test R²: {result['Test_R2']:.4f}")
    print(f"  CV R² (mean ± std): {result['CV_R2_Mean']:.4f} ± {result['CV_R2_Std']:.4f}")
    print(f"  Test RMSE: {result['Test_RMSE']:.4f}")
    print(f"  Test MAE: {result['Test_MAE']:.4f}")

# Create results DataFrame
results_df = pd.DataFrame(results)

# ============================================================================
# STEP 4: Save Model Comparison Results
# ============================================================================
print("\n[STEP 4] Saving model comparison results...")

results_path = os.path.join(OUTPUT_DIR, 'ml_results.csv')
results_df.to_csv(results_path, index=False)
print(f"Results saved to: {results_path}")

# Print summary table
print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)
summary_cols = ['Model', 'Test_R2', 'CV_R2_Mean', 'CV_R2_Std', 'Test_RMSE', 'Test_MAE']
print(results_df[summary_cols].to_string(index=False))

# ============================================================================
# STEP 5: Feature Importance Analysis
# ============================================================================
print("\n[STEP 5] Analyzing feature importance...")

# Get feature importance from tree-based models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest Feature Importance
rf_model = models['Random Forest']
rf_importance = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], color='steelblue')
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('Random Forest Feature Importance')
axes[0].grid(axis='x', alpha=0.3)

# Gradient Boosting Feature Importance
gb_model = models['Gradient Boosting']
gb_importance = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=True)

axes[1].barh(gb_importance['Feature'], gb_importance['Importance'], color='darkorange')
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('Gradient Boosting Feature Importance')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
feature_importance_path = os.path.join(FIGURES_DIR, 'feature_importance.png')
plt.savefig(feature_importance_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Feature importance plot saved to: {feature_importance_path}")

# Print feature importance
print("\nFeature Importance (Random Forest):")
for _, row in rf_importance.sort_values('Importance', ascending=False).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# ============================================================================
# STEP 6: SHAP Analysis
# ============================================================================
print("\n[STEP 6] SHAP analysis...")

if SHAP_AVAILABLE:
    try:
        # Use the best performing model (Random Forest or Gradient Boosting)
        best_model_name = results_df.loc[results_df['Test_R2'].idxmax(), 'Model']
        print(f"Using best model for SHAP: {best_model_name}")

        if best_model_name in ['Random Forest', 'Gradient Boosting']:
            best_model = models[best_model_name]
            X_explain = X_train  # Tree-based models use original features

            # Create SHAP explainer
            print("Creating SHAP explainer...")
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_explain)

            # Create SHAP summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_explain, feature_names=FEATURE_COLS, show=False)
            shap_path = os.path.join(FIGURES_DIR, 'shap_summary.png')
            plt.savefig(shap_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"SHAP summary plot saved to: {shap_path}")

            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = pd.DataFrame({
                'Feature': FEATURE_COLS,
                'Mean_SHAP': mean_shap
            }).sort_values('Mean_SHAP', ascending=False)

            print("\nSHAP Feature Importance:")
            for _, row in shap_importance.iterrows():
                print(f"  {row['Feature']}: {row['Mean_SHAP']:.4f}")
        else:
            print("Best model is not tree-based, using Random Forest for SHAP analysis")
            best_model = models['Random Forest']
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_train)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_train, feature_names=FEATURE_COLS, show=False)
            shap_path = os.path.join(FIGURES_DIR, 'shap_summary.png')
            plt.savefig(shap_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"SHAP summary plot saved to: {shap_path}")

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        # Create placeholder SHAP plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'SHAP analysis not available\n(Install shap package for detailed interpretation)',
                 ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.title('SHAP Summary Plot (Not Available)')
        shap_path = os.path.join(FIGURES_DIR, 'shap_summary.png')
        plt.savefig(shap_path, dpi=150, bbox_inches='tight')
        plt.close()
else:
    print("SHAP package not available, creating placeholder plot...")
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, 'SHAP analysis not available\n(Install shap package for detailed interpretation)',
             ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
    plt.title('SHAP Summary Plot (Not Available)')
    shap_path = os.path.join(FIGURES_DIR, 'shap_summary.png')
    plt.savefig(shap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Placeholder SHAP plot saved to: {shap_path}")

# ============================================================================
# STEP 7: Create Model Summary Report
# ============================================================================
print("\n[STEP 7] Creating model summary report...")

summary_path = os.path.join(OUTPUT_DIR, 'ml_model_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("QSPR ML MODEL SUMMARY REPORT\n")
    f.write("Target: Delta_PCE (PCE with modulator - PCE without modulator)\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATASET INFORMATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total samples: {len(df_clean)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"Features: {', '.join(FEATURE_COLS)}\n\n")

    f.write("TARGET STATISTICS (Delta_PCE)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Mean: {df_clean[TARGET_COL].mean():.4f}\n")
    f.write(f"Std: {df_clean[TARGET_COL].std():.4f}\n")
    f.write(f"Min: {df_clean[TARGET_COL].min():.4f}\n")
    f.write(f"Max: {df_clean[TARGET_COL].max():.4f}\n")
    f.write(f"Median: {df_clean[TARGET_COL].median():.4f}\n\n")

    f.write("MODEL PERFORMANCE COMPARISON\n")
    f.write("-" * 40 + "\n")
    f.write(results_df[summary_cols].to_string(index=False))
    f.write("\n\n")

    # Find best model
    best_idx = results_df['Test_R2'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    f.write(f"BEST MODEL: {best_model_name}\n")
    f.write(f"  Test R²: {results_df.loc[best_idx, 'Test_R2']:.4f}\n")
    f.write(f"  CV R²: {results_df.loc[best_idx, 'CV_R2_Mean']:.4f} ± {results_df.loc[best_idx, 'CV_R2_Std']:.4f}\n")
    f.write(f"  Test RMSE: {results_df.loc[best_idx, 'Test_RMSE']:.4f}\n")
    f.write(f"  Test MAE: {results_df.loc[best_idx, 'Test_MAE']:.4f}\n\n")

    f.write("FEATURE IMPORTANCE (Random Forest)\n")
    f.write("-" * 40 + "\n")
    for _, row in rf_importance.sort_values('Importance', ascending=False).iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\n")

    f.write("FILES GENERATED\n")
    f.write("-" * 40 + "\n")
    f.write(f"Processed data: {PROCESSED_DATA_PATH}\n")
    f.write(f"Model results: {results_path}\n")
    f.write(f"Feature importance plot: {feature_importance_path}\n")
    f.write(f"SHAP summary plot: {shap_path}\n")
    f.write(f"This summary: {summary_path}\n")

print(f"Model summary saved to: {summary_path}")

print("\n" + "=" * 80)
print("ML MODEL BUILDING COMPLETE")
print("=" * 80)
print(f"\nBest performing model: {results_df.loc[results_df['Test_R2'].idxmax(), 'Model']}")
print(f"Best Test R²: {results_df['Test_R2'].max():.4f}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
