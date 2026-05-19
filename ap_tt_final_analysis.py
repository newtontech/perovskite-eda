#!/usr/bin/env python3
"""
Atom Pair (AP) and Topological Torsion (TT) Fingerprint Analysis for QSPR
Comprehensive analysis for Delta_PCE prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("Atom Pair (AP) and Topological Torsion (TT) Fingerprint Analysis for QSPR")
print("=" * 80)
print()

# ============================================================================
# 1. Load Data
# ============================================================================
print("1. Loading data...")
data_path = '/share/yhm/test/AutoML_EDA/processed_data.csv'
df = pd.read_csv(data_path)

# Filter to valid SMILES and non-null Delta_PCE
df = df.dropna(subset=['smiles', 'Delta_PCE'])
df = df[df['smiles'].str.len() > 0]

# Remove duplicates
df = df.drop_duplicates(subset=['smiles'], keep='first')

print(f"   Loaded {len(df)} compounds with valid SMILES and Delta_PCE")
print(f"   Delta_PCE range: [{df['Delta_PCE'].min():.2f}, {df['Delta_PCE'].max():.2f}]")
print(f"   Delta_PCE mean ± std: {df['Delta_PCE'].mean():.2f} ± {df['Delta_PCE'].std():.2f}")
print()

# ============================================================================
# 2. Generate Fingerprints
# ============================================================================
print("2. Generating molecular fingerprints...")

def generate_atom_pair_fingerprint(mol, nBits=2048):
    """Generate Atom Pair fingerprint"""
    try:
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)
        arr = np.zeros(nBits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(nBits, dtype=np.int8)

def generate_topological_torsion_fingerprint(mol, nBits=2048):
    """Generate Topological Torsion fingerprint"""
    try:
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits)
        arr = np.zeros(nBits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(nBits, dtype=np.int8)

# Convert SMILES to molecules
valid_mols = []
valid_indices = []
for idx, row in df.iterrows():
    try:
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None and mol.GetNumAtoms() > 0:
            valid_mols.append(mol)
            valid_indices.append(idx)
    except:
        pass

print(f"   Successfully parsed {len(valid_mols)} molecules")
df_valid = df.loc[valid_indices].copy()
y = df_valid['Delta_PCE'].values

# Generate fingerprints
nBits = 2048
ap_fps = np.array([generate_atom_pair_fingerprint(mol, nBits) for mol in valid_mols])
tt_fps = np.array([generate_topological_torsion_fingerprint(mol, nBits) for mol in valid_mols])

print(f"   Generated Atom Pair fingerprints: {ap_fps.shape}")
print(f"   Generated Topological Torsion fingerprints: {tt_fps.shape}")
print(f"   AP non-zero bits per molecule: {np.mean(np.sum(ap_fps > 0, axis=1)):.1f}")
print(f"   TT non-zero bits per molecule: {np.mean(np.sum(tt_fps > 0, axis=1)):.1f}")
print()

# Save fingerprints
print("   Saving fingerprint data...")
np.save('/share/yhm/test/AutoML_EDA/fingerprints/atompair_fingerprints.npy', ap_fps)
np.save('/share/yhm/test/AutoML_EDA/fingerprints/torsion_fingerprints.npy', tt_fps)
print("   Fingerprints saved successfully")
print()

# ============================================================================
# 3. Fingerprint Comparison Analysis
# ============================================================================
print("3. Comparative analysis of AP vs TT fingerprints...")

# Calculate bit statistics
ap_bit_freq = np.mean(ap_fps, axis=0)
tt_bit_freq = np.mean(tt_fps, axis=0)

# Calculate correlations with Delta_PCE
ap_bit_corr = np.array([np.corrcoef(ap_fps[:, i], y)[0, 1] if np.std(ap_fps[:, i]) > 0 else 0
                        for i in range(nBits)])
tt_bit_corr = np.array([np.corrcoef(tt_fps[:, i], y)[0, 1] if np.std(tt_fps[:, i]) > 0 else 0
                        for i in range(nBits)])

# Handle NaN values
ap_bit_corr = np.nan_to_num(ap_bit_corr, nan=0.0)
tt_bit_corr = np.nan_to_num(tt_bit_corr, nan=0.0)

# Find significant bits
threshold = 0.15
ap_significant = np.where(np.abs(ap_bit_corr) > threshold)[0]
tt_significant = np.where(np.abs(tt_bit_corr) > threshold)[0]
overlap_bits = np.intersect1d(ap_significant, tt_significant)

print(f"   AP significant bits (|r| > {threshold}): {len(ap_significant)}")
print(f"   TT significant bits (|r| > {threshold}): {len(tt_significant)}")
print(f"   Overlapping significant bits: {len(overlap_bits)}")
print()

# Create visualization figures
fig = plt.figure(figsize=(16, 12))

# Bit frequency distributions
ax1 = plt.subplot(2, 3, 1)
ax1.hist(ap_bit_freq[ap_bit_freq > 0], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Bit Frequency', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('AP Bit Frequency Distribution', fontsize=12, fontweight='bold')

ax2 = plt.subplot(2, 3, 2)
ax2.hist(tt_bit_freq[tt_bit_freq > 0], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Bit Frequency', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('TT Bit Frequency Distribution', fontsize=12, fontweight='bold')

# Bit correlation distributions
ax3 = plt.subplot(2, 3, 4)
ax3.hist(ap_bit_corr, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax3.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold (±{threshold})')
ax3.axvline(x=-threshold, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Correlation with Delta_PCE', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('AP Bit Correlation Distribution', fontsize=12, fontweight='bold')
ax3.legend()

ax4 = plt.subplot(2, 3, 5)
ax4.hist(tt_bit_corr, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
ax4.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold (±{threshold})')
ax4.axvline(x=-threshold, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Correlation with Delta_PCE', fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('TT Bit Correlation Distribution', fontsize=12, fontweight='bold')
ax4.legend()

# Venn diagram
try:
    from matplotlib_venn import venn2
    ax5 = plt.subplot(2, 3, 3)
    ap_only = len(ap_significant) - len(overlap_bits)
    tt_only = len(tt_significant) - len(overlap_bits)
    venn2(subsets=(ap_only, tt_only, len(overlap_bits)),
          set_labels=('Atom Pair', 'Topological Torsion'),
          set_colors=('#3498db', '#e74c3c'), alpha=0.7, ax=ax5)
    ax5.set_title(f'Significant Bits Overlap (|r| > {threshold})', fontsize=12, fontweight='bold')
except ImportError:
    pass

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/ap_tt_analysis_overview.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. Dimensionality Reduction
# ============================================================================
print("4. Dimensionality reduction analysis...")

# Filter out constant features
ap_var = np.var(ap_fps, axis=0)
tt_var = np.var(tt_fps, axis=0)
ap_active = ap_var > 0
tt_active = tt_var > 0

ap_fps_filtered = ap_fps[:, ap_active]
tt_fps_filtered = tt_fps[:, tt_active]

print(f"   AP active bits: {np.sum(ap_active)}")
print(f"   TT active bits: {np.sum(tt_active)}")

# PCA
n_comp = min(50, ap_fps_filtered.shape[1], tt_fps_filtered.shape[1])
pca_ap = PCA(n_components=n_comp)
pca_tt = PCA(n_components=n_comp)

ap_pca = pca_ap.fit_transform(ap_fps_filtered)
tt_pca = pca_tt.fit_transform(tt_fps_filtered)

print(f"   AP PCA explained variance (first 10 PCs): {np.sum(pca_ap.explained_variance_ratio_[:10]):.2%}")
print(f"   TT PCA explained variance (first 10 PCs): {np.sum(pca_tt.explained_variance_ratio_[:10]):.2%}")
print()

# PCA plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

scatter_ap = axes[0].scatter(ap_pca[:, 0], ap_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel(f'PC1 ({pca_ap.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
axes[0].set_ylabel(f'PC2 ({pca_ap.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
axes[0].set_title('Atom Pair PCA Projection', fontsize=14, fontweight='bold')
plt.colorbar(scatter_ap, ax=axes[0], label='Delta_PCE')

scatter_tt = axes[1].scatter(tt_pca[:, 0], tt_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[1].set_xlabel(f'PC1 ({pca_tt.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
axes[1].set_ylabel(f'PC2 ({pca_tt.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
axes[1].set_title('Topological Torsion PCA Projection', fontsize=14, fontweight='bold')
plt.colorbar(scatter_tt, ax=axes[1], label='Delta_PCE')

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/ap_tt_pca_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. Machine Learning Modeling
# ============================================================================
print("5. Machine learning modeling...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(X, y, model_name, model):
    """Evaluate model using cross-validation"""
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))
    mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    return {
        'Model': model_name,
        'R2_mean': np.mean(r2_scores),
        'R2_std': np.std(r2_scores),
        'RMSE_mean': np.mean(rmse_scores),
        'RMSE_std': np.std(rmse_scores),
        'MAE_mean': np.mean(mae_scores),
        'MAE_std': np.std(mae_scores)
    }

# Train models
rf_ap = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_tt = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_combined = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)

ap_tt_combined = np.hstack([ap_fps_filtered, tt_fps_filtered])

results = []
print("   Training Random Forest models...")
results.append(evaluate_model(ap_fps_filtered, y, 'Random Forest (AP)', rf_ap))
results.append(evaluate_model(tt_fps_filtered, y, 'Random Forest (TT)', rf_tt))
results.append(evaluate_model(ap_tt_combined, y, 'Random Forest (AP+TT)', rf_combined))

print("   Training Gradient Boosting models...")
gb_ap = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb_tt = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb_combined = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

results.append(evaluate_model(ap_fps_filtered, y, 'Gradient Boosting (AP)', gb_ap))
results.append(evaluate_model(tt_fps_filtered, y, 'Gradient Boosting (TT)', gb_tt))
results.append(evaluate_model(ap_tt_combined, y, 'Gradient Boosting (AP+TT)', gb_combined))

results_df = pd.DataFrame(results)

print("   Model Performance Summary:")
for _, row in results_df.iterrows():
    print(f"   {row['Model']}: R² = {row['R2_mean']:.4f} ± {row['R2_std']:.4f}, RMSE = {row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}")
print()

# Save ML results
results_df.to_csv('/share/yhm/test/AutoML_EDA/fingerprints/ap_tt_ml_results.csv', index=False)

# Performance comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = ['R2_mean', 'RMSE_mean', 'MAE_mean']
metric_labels = ['R² Score', 'RMSE', 'MAE']
colors = ['#3498db', '#e74c3c', '#9b59b6', '#3498db', '#e74c3c', '#9b59b6']
hatch_patterns = ['//', None, None, '\\\\', None, None]

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    values = results_df[metric].values
    errors = results_df[metric.replace('mean', 'std')].values
    bars = axes[idx].bar(range(len(results_df)), values, yerr=errors, color=colors, alpha=0.7, capsize=5)
    for bar, hatch in zip(bars, hatch_patterns):
        if hatch:
            bar.set_hatch(hatch)
    axes[idx].set_xticks(range(len(results_df)))
    axes[idx].set_xticklabels(results_df['Model'].values, rotation=45, ha='right')
    axes[idx].set_ylabel(label, fontsize=12)
    axes[idx].set_title(f'{label} by Model', fontsize=14, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/ap_tt_ml_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. Feature Importance Analysis
# ============================================================================
print("6. Feature importance analysis...")

# Train final models
rf_ap.fit(ap_fps_filtered, y)
rf_tt.fit(tt_fps_filtered, y)

# Get top features
ap_importance = rf_ap.feature_importances_
tt_importance = rf_tt.feature_importances_

ap_original_indices = np.where(ap_active)[0]
tt_original_indices = np.where(tt_active)[0]

ap_top20_idx = ap_original_indices[np.argsort(ap_importance)[-20:][::-1]]
tt_top20_idx = tt_original_indices[np.argsort(tt_importance)[-20:][::-1]]

ap_features = pd.DataFrame({
    'Bit_Index': ap_top20_idx,
    'Importance': np.sort(ap_importance)[-20:][::-1],
    'Frequency': ap_bit_freq[ap_top20_idx],
    'Correlation': ap_bit_corr[ap_top20_idx]
})

tt_features = pd.DataFrame({
    'Bit_Index': tt_top20_idx,
    'Importance': np.sort(tt_importance)[-20:][::-1],
    'Frequency': tt_bit_freq[tt_top20_idx],
    'Correlation': tt_bit_corr[tt_top20_idx]
})

print("   Top 10 AP bits:")
for i in range(min(10, len(ap_features))):
    print(f"     Bit {int(ap_features['Bit_Index'].iloc[i])}: Importance={ap_features['Importance'].iloc[i]:.4f}, Freq={ap_features['Frequency'].iloc[i]:.4f}, Corr={ap_features['Correlation'].iloc[i]:.4f}")

print()
print("   Top 10 TT bits:")
for i in range(min(10, len(tt_features))):
    print(f"     Bit {int(tt_features['Bit_Index'].iloc[i])}: Importance={tt_features['Importance'].iloc[i]:.4f}, Freq={tt_features['Frequency'].iloc[i]:.4f}, Corr={tt_features['Correlation'].iloc[i]:.4f}")
print()

# Feature importance plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].barh(range(20), ap_features['Importance'].values, color='#3498db', alpha=0.7, edgecolor='black')
axes[0].set_yticks(range(20))
axes[0].set_yticklabels([f"Bit {int(idx)}" for idx in ap_features['Bit_Index'].values])
axes[0].set_xlabel('Feature Importance', fontsize=12)
axes[0].set_title('Top 20 Atom Pair Bits', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

axes[1].barh(range(20), tt_features['Importance'].values, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(20))
axes[1].set_yticklabels([f"Bit {int(idx)}" for idx in tt_features['Bit_Index'].values])
axes[1].set_xlabel('Feature Importance', fontsize=12)
axes[1].set_title('Top 20 Topological Torsion Bits', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/ap_tt_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Save comparison
comparison_data = []
for i in range(20):
    comparison_data.append({
        'Type': 'AP',
        'Bit_Index': int(ap_features['Bit_Index'].iloc[i]),
        'Importance': ap_features['Importance'].iloc[i],
        'Frequency': ap_features['Frequency'].iloc[i],
        'Correlation': ap_features['Correlation'].iloc[i]
    })
for i in range(20):
    comparison_data.append({
        'Type': 'TT',
        'Bit_Index': int(tt_features['Bit_Index'].iloc[i]),
        'Importance': tt_features['Importance'].iloc[i],
        'Frequency': tt_features['Frequency'].iloc[i],
        'Correlation': tt_features['Correlation'].iloc[i]
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('/share/yhm/test/AutoML_EDA/fingerprints/ap_tt_comparison.csv', index=False)

# ============================================================================
# 7. Generate Summary Report
# ============================================================================
print("7. Generating summary report...")

report = f"""
{'=' * 80}
ATOM PAIR (AP) AND TOPOLOGICAL TORSION (TT) FINGERPRINT ANALYSIS
QSPR Study for Delta_PCE Prediction
{'=' * 80}

DATASET SUMMARY
--------------
Total compounds analyzed: {len(df_valid)}
Delta_PCE range: [{df_valid['Delta_PCE'].min():.2f}, {df_valid['Delta_PCE'].max():.2f}]
Delta_PCE mean ± std: {df_valid['Delta_PCE'].mean():.2f} ± {df_valid['Delta_PCE'].std():.2f}

FINGERPRINT GENERATION
---------------------
Fingerprint size: {nBits} bits
Atom Pair (AP) fingerprints: {ap_fps.shape}
Topological Torsion (TT) fingerprints: {tt_fps.shape}

AP non-zero bits per molecule: {np.mean(np.sum(ap_fps > 0, axis=1)):.1f} ± {np.std(np.sum(ap_fps > 0, axis=1)):.1f}
TT non-zero bits per molecule: {np.mean(np.sum(tt_fps > 0, axis=1)):.1f} ± {np.std(np.sum(tt_fps > 0, axis=1)):.1f}

AP active bits (non-zero variance): {np.sum(ap_active)}/{nBits} ({np.sum(ap_active)/nBits:.1%})
TT active bits (non-zero variance): {np.sum(tt_active)}/{nBits} ({np.sum(tt_active)/nBits:.1%})

BIT ANALYSIS
-----------
AP significant bits (|r| > {threshold}): {len(ap_significant)}
TT significant bits (|r| > {threshold}): {len(tt_significant)}
Overlapping significant bits: {len(overlap_bits)}

Highest AP bit correlation: {np.abs(ap_bit_corr).max():.4f}
Highest TT bit correlation: {np.abs(tt_bit_corr).max():.4f}

DIMENSIONALITY REDUCTION
----------------------
AP PCA explained variance (first 10 PCs): {np.sum(pca_ap.explained_variance_ratio_[:10]):.2%}
TT PCA explained variance (first 10 PCs): {np.sum(pca_tt.explained_variance_ratio_[:10]):.2%}

MACHINE LEARNING RESULTS
-----------------------
"""

for _, row in results_df.iterrows():
    report += f"""
{row['Model']}:
  R² Score: {row['R2_mean']:.4f} ± {row['R2_std']:.4f}
  RMSE: {row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}
  MAE: {row['MAE_mean']:.4f} ± {row['MAE_std']:.4f}
"""

report += f"""

BEST PERFORMING MODEL
--------------------
Model: {results_df.loc[results_df['R2_mean'].idxmax(), 'Model']}
R² Score: {results_df['R2_mean'].max():.4f} ± {results_df.loc[results_df['R2_mean'].idxmax(), 'R2_std']:.4f}

TOP FEATURES (ATOM PAIR)
-----------------------
"""
for i in range(min(10, len(ap_features))):
    report += f"""{i+1:2d}. Bit {int(ap_features['Bit_Index'].iloc[i])} - Importance: {ap_features['Importance'].iloc[i]:.4f}, Correlation: {ap_features['Correlation'].iloc[i]:.4f}
"""

report += f"""

TOP FEATURES (TOPOLOGICAL TORSION)
---------------------------------
"""
for i in range(min(10, len(tt_features))):
    report += f"""{i+1:2d}. Bit {int(tt_features['Bit_Index'].iloc[i])} - Importance: {tt_features['Importance'].iloc[i]:.4f}, Correlation: {tt_features['Correlation'].iloc[i]:.4f}
"""

report += f"""

KEY FINDINGS
-----------
1. Fingerprint Comparison:
   - AP captures {np.sum(ap_active)} unique bits, TT captures {np.sum(tt_active)} unique bits
   - {len(overlap_bits)} bits are significant for both methods (|r| > {threshold})
   - TT fingerprints show {'higher' if np.mean(np.sum(tt_fps > 0, axis=1)) > np.mean(np.sum(ap_fps > 0, axis=1)) else 'lower'} bit density

2. Dimensionality Reduction:
   - AP: {np.sum(pca_ap.explained_variance_ratio_[:10]):.1%} variance in first 10 PCs
   - TT: {np.sum(pca_tt.explained_variance_ratio_[:10]):.1%} variance in first 10 PCs
   - {'AP' if np.sum(pca_ap.explained_variance_ratio_[:10]) > np.sum(pca_tt.explained_variance_ratio_[:10]) else 'TT'} fingerprints show better compressibility

3. ML Performance:
   - Best model: {results_df.loc[results_df['R2_mean'].idxmax(), 'Model']}
   - R² Score: {results_df['R2_mean'].max():.4f}
   - {'AP' if results_df.iloc[0]['R2_mean'] > results_df.iloc[1]['R2_mean'] else 'TT'} fingerprints show stronger correlation with Delta_PCE

CHEMICAL INTERPRETATION
----------------------
Atom Pair (AP) Fingerprints:
- Encode distances between pairs of atoms
- Each bit represents atom type combinations at specific topological distances
- Better for capturing local molecular environments
- More interpretable for functional group analysis

Topological Torsion (TT) Fingerprints:
- Encode sequences of four consecutive bonded atoms
- Capture torsional angle patterns along molecular paths
- Better for capturing 3D molecular shape and flexibility
- More sensitive to stereochemistry

OUTPUT FILES
----------
Fingerprint Data:
- /share/yhm/test/AutoML_EDA/fingerprints/atompair_fingerprints.npy
- /share/yhm/test/AutoML_EDA/fingerprints/torsion_fingerprints.npy

Analysis Results:
- /share/yhm/test/AutoML_EDA/fingerprints/ap_tt_comparison.csv
- /share/yhm/test/AutoML_EDA/fingerprints/ap_tt_ml_results.csv

Figures:
- /share/yhm/test/AutoML_EDA/figures/ap_tt_analysis_overview.png
- /share/yhm/test/AutoML_EDA/figures/ap_tt_pca_scatter.png
- /share/yhm/test/AutoML_EDA/figures/ap_tt_ml_comparison.png
- /share/yhm/test/AutoML_EDA/figures/ap_tt_feature_importance.png

Report:
- /share/yhm/test/AutoML_EDA/fingerprints/ap_tt_report.txt

{'=' * 80}
Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""

with open('/share/yhm/test/AutoML_EDA/fingerprints/ap_tt_report.txt', 'w') as f:
    f.write(report)

print(report)
print("\nAll analysis complete! Results saved.")
