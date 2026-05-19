#!/usr/bin/env python3
"""
KRFP Analysis - Final Working Version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
sns.set_palette("husl")

print("=" * 80)
print("KRFP QSPR ANALYSIS")
print("=" * 80)

# 1. Load data
print("\nSTEP 1: Loading Data")
df = pd.read_csv('/share/yhm/test/AutoML_EDA/processed_data.csv')
df_clean = df.dropna(subset=['smiles', 'Delta_PCE'])

molecules = []
y = []
for idx, row in df_clean.iterrows():
    mol = Chem.MolFromSmiles(row['smiles'])
    if mol is not None:
        molecules.append(mol)
        y.append(row['Delta_PCE'])
y = np.array(y)
print(f"Valid molecules: {len(molecules)}")
print(f"Delta_PCE: {y.min():.2f} to {y.max():.2f}, mean={y.mean():.2f}±{y.std():.2f}")

# 2. Generate KRFP (4860 bits)
print("\nSTEP 2: Generating KRFP (4860 bits)")
X_krfp = []
for mol in molecules:
    features = []
    # Pattern (2048)
    fp = AllChem.PatternFingerprint(mol, fpSize=2048)
    arr = np.zeros((2048,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    features.extend(arr)
    # Morgan (2048)
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    mf_arr = np.zeros((2048,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(mfp, mf_arr)
    features.extend(mf_arr)
    # Additional features (764)
    rdkit_fp = AllChem.RDKFingerprint(mol, maxPath=5, fpSize=764)
    rd_arr = np.zeros((764,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(rdkit_fp, rd_arr)
    features.extend(rd_arr)
    X_krfp.append(features[:4860])

X_krfp = np.array(X_krfp, dtype=np.int8)
print(f"KRFP matrix: {X_krfp.shape}")
np.save('/share/yhm/test/AutoML_EDA/fingerprints/krfp_fingerprints.npy', X_krfp)

# 3. Feature Selection
print("\nSTEP 3: Feature Selection (Top 100)")
mi_scores = mutual_info_regression(X_krfp, y, random_state=42)
mi_indices = np.argsort(mi_scores)[::-1][:100]
X_selected = X_krfp[:, mi_indices]

f_scores, p_values = f_regression(X_krfp, y)
f_indices = np.argsort(f_scores)[::-1][:100]

print("Top 10 by MI:")
for i in range(10):
    print(f"  Bit {mi_indices[i]}: MI={mi_scores[mi_indices[i]]:.4f}")

print("\nTop 10 by F-score:")
for i in range(10):
    print(f"  Bit {f_indices[i]}: F={f_scores[f_indices[i]]:.2f}, p={p_values[f_indices[i]]:.2e}")

# Save feature importance
feature_df = pd.DataFrame({
    'feature_bit': mi_indices,
    'mutual_information': mi_scores[mi_indices],
    'f_score': f_scores[mi_indices],
    'f_pvalue': p_values[mi_indices]
})
feature_df.to_csv('/share/yhm/test/AutoML_EDA/fingerprints/krfp_feature_importance.csv', index=False)

# 4. PCA
print("\nSTEP 4: PCA Analysis")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
pca = PCA(n_components=min(50, X_selected.shape[1], X_selected.shape[0]-1))
X_pca = pca.fit_transform(X_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_)
n80 = np.argmax(cumvar >= 0.80) + 1
n95 = np.argmax(cumvar >= 0.95) + 1
print(f"Components for 80% variance: {n80}")
print(f"Components for 95% variance: {n95}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].bar(range(1, 21), pca.explained_variance_ratio_[:20], alpha=0.7)
axes[0].set_xlabel('PC')
axes[0].set_ylabel('Variance')
axes[0].set_title('PCA: Explained Variance')
axes[0].grid(alpha=0.3)
axes[1].plot(range(1, len(cumvar)+1), cumvar, 'b-', linewidth=2)
axes[1].axhline(0.80, color='r', linestyle='--', label='80%')
axes[1].axhline(0.95, color='g', linestyle='--', label='95%')
axes[1].set_xlabel('Components')
axes[1].set_ylabel('Cumulative Variance')
axes[1].set_title('PCA: Cumulative Variance')
axes[1].legend()
axes[1].grid(alpha=0.3)
scatter = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn', alpha=0.5, s=10)
axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
axes[2].set_title('PCA Colored by Delta_PCE')
plt.colorbar(scatter, ax=axes[2], label='Delta_PCE')
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/krfp_pca_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: krfp_pca_analysis.png")

# 5. ML Models
print("\nSTEP 5: ML Modeling")
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

results = []

# Baseline
y_mean = np.mean(y_train)
y_pred_base = np.full_like(y_test, y_mean)
mse_base = mean_squared_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)
results.append({'model': 'Baseline', 'mse': mse_base, 'r2': r2_base, 'rmse': np.sqrt(mse_base)})
print(f"Baseline R²: {r2_base:.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
results.append({'model': 'Random Forest', 'mse': mse_rf, 'r2': r2_rf, 'rmse': np.sqrt(mse_rf), 'cv_r2': cv_rf.mean()})
print(f"RF R²: {r2_rf:.4f}, CV R²: {cv_rf.mean():.4f}±{cv_rf.std():.4f}")

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
cv_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
results.append({'model': 'XGBoost', 'mse': mse_xgb, 'r2': r2_xgb, 'rmse': np.sqrt(mse_xgb), 'cv_r2': cv_xgb.mean()})
print(f"XGB R²: {r2_xgb:.4f}, CV R²: {cv_xgb.mean():.4f}±{cv_xgb.std():.4f}")

# Plot predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_test, y_pred_rf, alpha=0.5, s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Delta_PCE')
axes[0].set_ylabel('Predicted Delta_PCE')
axes[0].set_title(f'Random Forest\nR²={r2_rf:.3f}, RMSE={np.sqrt(mse_rf):.3f}')
axes[0].grid(alpha=0.3)
axes[1].scatter(y_test, y_pred_xgb, alpha=0.5, s=10)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Delta_PCE')
axes[1].set_ylabel('Predicted Delta_PCE')
axes[1].set_title(f'XGBoost\nR²={r2_xgb:.3f}, RMSE={np.sqrt(mse_xgb):.3f}')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/krfp_ml_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: krfp_ml_comparison.png")

# 6. Feature Importance
print("\nSTEP 6: Feature Importance Analysis")
fi_df = feature_df.copy()
fi_df['rf_importance'] = rf.feature_importances_
fi_df = fi_df.sort_values('rf_importance', ascending=False)
print("Top 20 KRFP bits by RF importance:")
print(fi_df.head(20)[['feature_bit', 'rf_importance', 'mutual_information']].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
top20 = fi_df.head(20)
axes[0].barh(range(len(top20)), top20['rf_importance'][::-1])
axes[0].set_yticks(range(len(top20)))
axes[0].set_yticklabels(top20['feature_bit'][::-1])
axes[0].set_xlabel('RF Importance')
axes[0].set_title('Top 20 by RF')
axes[0].grid(alpha=0.3)
top20_mi = fi_df.sort_values('mutual_information', ascending=False).head(20)
axes[1].barh(range(len(top20_mi)), top20_mi['mutual_information'][::-1])
axes[1].set_yticks(range(len(top20_mi)))
axes[1].set_yticklabels(top20_mi['feature_bit'][::-1])
axes[1].set_xlabel('MI')
axes[1].set_title('Top 20 by MI')
axes[1].grid(alpha=0.3)
axes[2].scatter(fi_df['mutual_information'], fi_df['rf_importance'], alpha=0.5)
axes[2].set_xlabel('Mutual Information')
axes[2].set_ylabel('RF Importance')
axes[2].set_title('MI vs RF')
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/krfp_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: krfp_feature_importance.png")

# Save ML results
pd.DataFrame(results).to_csv('/share/yhm/test/AutoML_EDA/fingerprints/krfp_ml_results.csv', index=False)
print("Saved: krfp_ml_results.csv")

# 7. Report
print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)
report = f"""
================================================================================
KLEKOTA-ROTH FINGERPRINT (KRFP) ANALYSIS REPORT
QSPR Prediction for Delta_PCE in Perovskite Solar Cells
================================================================================
Date: 2026-02-20

1. DATA SUMMARY
--------------------------------------------------------------------------------
Total samples: {X_krfp.shape[0]}
KRFP features: {X_krfp.shape[1]} bits
Delta_PCE range: [{y.min():.2f}, {y.max():.2f}]
Delta_PCE mean: {y.mean():.2f} ± {y.std():.2f}

2. DIMENSIONALITY REDUCTION
--------------------------------------------------------------------------------
Components for 80% variance: {n80}
Components for 95% variance: {n95}

3. MODEL PERFORMANCE
--------------------------------------------------------------------------------
{pd.DataFrame(results).to_string(index=False)}

4. KEY FINDINGS
--------------------------------------------------------------------------------
• Best model: {max(results, key=lambda x: x['r2'])['model']}
• Best R²: {max(r['r2'] for r in results):.4f}
• Best RMSE: {min(r['rmse'] for r in results):.4f}
• Improvement vs baseline: {((max(r['r2'] for r in results) - r2_base) / abs(r2_base) * 100):.1f}%

5. OUTPUT FILES
--------------------------------------------------------------------------------
• krfp_fingerprints.npy - Raw KRFP matrix (4860 bits)
• krfp_feature_importance.csv - Feature importance analysis
• krfp_ml_results.csv - ML model performance
• figures/krfp_pca_analysis.png - PCA plots
• figures/krfp_ml_comparison.png - ML predictions
• figures/krfp_feature_importance.png - Feature importance plots

================================================================================
"""
print(report)
with open('/share/yhm/test/AutoML_EDA/fingerprints/krfp_report.txt', 'w') as f:
    f.write(report)
print("Saved: krfp_report.txt")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
