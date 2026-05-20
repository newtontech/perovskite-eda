#!/usr/bin/env python3
"""
Klekota-Roth Fingerprints (KRFP) Analysis for QSPR - Optimized Version
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import warnings
import gc

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 70)
print("Klekota-Roth Fingerprints (KRFP) Analysis")
print("=" * 70)

# 1. Load Data
print("\n[1] Loading data...")
df = pd.read_excel('/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx')
print(f"    Loaded {len(df)} samples")

# Convert PCE columns to numeric
df['jv_reverse_scan_pce'] = pd.to_numeric(df['jv_reverse_scan_pce'], errors='coerce')
df['jv_reverse_scan_pce_without_modulator'] = pd.to_numeric(df['jv_reverse_scan_pce_without_modulator'], errors='coerce')

# Calculate Delta_PCE
df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']

# Filter valid
valid_mask = df['smiles'].notna() & df['Delta_PCE'].notna()
df_valid = df[valid_mask].copy()
print(f"    Valid samples: {len(df_valid)}")
print(f"    Delta_PCE range: [{df_valid['Delta_PCE'].min():.4f}, {df_valid['Delta_PCE'].max():.4f}]")

# 2. Generate KRFP (4860 bits)
print("\n[2] Generating 4860-bit KRFP fingerprints...")

def get_krfp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            mol, nBits=4860, minLength=1, maxLength=7, includeChirality=True
        )
        return list(fp)
    except:
        return None

fps = []
valid_idx = []
smiles_list = df_valid['smiles'].tolist()
delta_pce_list = df_valid['Delta_PCE'].tolist()

for i, smi in enumerate(smiles_list):
    fp = get_krfp(smi)
    if fp is not None:
        fps.append(fp)
        valid_idx.append(i)
    if (i+1) % 500 == 0:
        print(f"    Processed {i+1}/{len(smiles_list)}...")

fingerprints = np.array(fps, dtype=np.int8)
y = np.array([delta_pce_list[i] for i in valid_idx])
smiles_final = [smiles_list[i] for i in valid_idx]

print(f"    Generated fingerprints: {fingerprints.shape}")

# Clean up
del fps, smiles_list, delta_pce_list
gc.collect()

# 3. Feature Selection
print("\n[3] Feature selection (MI)...")
mi_scores = mutual_info_regression(fingerprints, y, random_state=42, n_neighbors=3)
top50_idx = np.argsort(mi_scores)[::-1][:50]
top50_scores = mi_scores[top50_idx]
print(f"    Top MI feature: Bit {top50_idx[0]} (MI={top50_scores[0]:.6f})")

# 4. PCA
print("\n[4] PCA...")
pca = PCA(n_components=50, random_state=42)
fp_pca = pca.fit_transform(fingerprints)
print(f"    PC1+PC2: {sum(pca.explained_variance_ratio_[:2])*100:.2f}%")

# 5. t-SNE
print("\n[5] t-SNE...")
perp = min(30, len(fingerprints)-1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perp, n_iter=500)
fp_tsne = tsne.fit_transform(fingerprints)
print(f"    t-SNE completed")

# 6. ML Models
print("\n[6] Training ML models...")
X_sel = fingerprints[:, top50_idx]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

# RF
print("    Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf_r2 = cross_val_score(rf, X_scaled, y, cv=cv, scoring='r2')
rf.fit(X_scaled, y)
results.append({'Model': 'RandomForest_50feat', 'R2': rf_r2.mean(), 'R2_std': rf_r2.std()})
print(f"      R2: {rf_r2.mean():.4f} (+/- {rf_r2.std():.4f})")

# XGBoost
if HAS_XGB:
    print("    XGBoost...")
    xgb_mod = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=6, verbosity=0)
    xgb_r2 = cross_val_score(xgb_mod, X_scaled, y, cv=cv, scoring='r2')
    xgb_mod.fit(X_scaled, y)
    results.append({'Model': 'XGBoost_50feat', 'R2': xgb_r2.mean(), 'R2_std': xgb_r2.std()})
    print(f"      R2: {xgb_r2.mean():.4f} (+/- {xgb_r2.std():.4f})")

# RF with PCA
print("    Random Forest (PCA 20)...")
rf_pca = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf_pca_r2 = cross_val_score(rf_pca, fp_pca[:, :20], y, cv=cv, scoring='r2')
results.append({'Model': 'RandomForest_PCA20', 'R2': rf_pca_r2.mean(), 'R2_std': rf_pca_r2.std()})
print(f"      R2: {rf_pca_r2.mean():.4f} (+/- {rf_pca_r2.std():.4f})")

results_df = pd.DataFrame(results)

# 7. Pattern Analysis
print("\n[7] Pattern analysis...")
median_y = np.median(y)
high_mask = y >= median_y
low_mask = y < median_y

prev_high = fingerprints[high_mask].mean(axis=0)
prev_low = fingerprints[low_mask].mean(axis=0)
diff = prev_high - prev_low

high_bits = np.argsort(diff)[::-1][:20]
low_bits = np.argsort(diff)[:20]

# 8. Save Results
print("\n[8] Saving results...")
out_dir = '/share/yhm/test/AutoML_EDA'

# Fingerprints
fp_df = pd.DataFrame(fingerprints)
fp_df.columns = [f'KRFP_{i}' for i in range(4860)]
fp_df.insert(0, 'smiles', smiles_final)
fp_df['Delta_PCE'] = y
fp_df.to_csv(f'{out_dir}/fingerprints/krfp_fingerprints.csv', index=False)
print("    Saved: krfp_fingerprints.csv")

# Feature importance
fi_df = pd.DataFrame({
    'rank': range(1, 51),
    'bit_index': top50_idx,
    'mutual_information': top50_scores
})
fi_df.to_csv(f'{out_dir}/fingerprints/krfp_feature_importance.csv', index=False)
print("    Saved: krfp_feature_importance.csv")

# ML results
results_df.to_csv(f'{out_dir}/fingerprints/krfp_ml_results.csv', index=False)
print("    Saved: krfp_ml_results.csv")

# PCA plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sc1 = ax[0].scatter(fp_pca[:, 0], fp_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6, s=20)
ax[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax[0].set_title('PCA of KRFP')
plt.colorbar(sc1, ax=ax[0], label='Delta_PCE')

ax[1].bar(range(1, 21), pca.explained_variance_ratio_[:20], alpha=0.7)
ax[1].plot(range(1, 21), np.cumsum(pca.explained_variance_ratio_[:20]), 'ro-')
ax[1].set_xlabel('PC')
ax[1].set_ylabel('Variance')
ax[1].set_title('PCA Explained Variance')
ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{out_dir}/figures/krfp_pca.png', dpi=200)
plt.close()
print("    Saved: krfp_pca.png")

# t-SNE plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sc2 = ax[0].scatter(fp_tsne[:, 0], fp_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.6, s=20)
ax[0].set_xlabel('t-SNE 1')
ax[0].set_ylabel('t-SNE 2')
ax[0].set_title('t-SNE of KRFP')
plt.colorbar(sc2, ax=ax[0], label='Delta_PCE')

colors = ['red' if h else 'blue' for h in high_mask]
ax[1].scatter(fp_tsne[:, 0], fp_tsne[:, 1], c=colors, alpha=0.5, s=15)
ax[1].set_xlabel('t-SNE 1')
ax[1].set_ylabel('t-SNE 2')
ax[1].set_title('t-SNE: High(red) vs Low(blue) Delta_PCE')
plt.tight_layout()
plt.savefig(f'{out_dir}/figures/krfp_tsne.png', dpi=200)
plt.close()
print("    Saved: krfp_tsne.png")

# Summary plot
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].barh(range(20), top50_scores[:20][::-1])
ax[0, 0].set_yticks(range(20))
ax[0, 0].set_yticklabels([f'Bit {top50_idx[i]}' for i in range(19, -1, -1)])
ax[0, 0].set_xlabel('MI Score')
ax[0, 0].set_title('Top 20 Fingerprint Bits')

x_pos = np.arange(10)
ax[0, 1].bar(x_pos - 0.2, prev_high[high_bits[:10]], 0.4, label='High', color='red', alpha=0.7)
ax[0, 1].bar(x_pos + 0.2, prev_low[high_bits[:10]], 0.4, label='Low', color='blue', alpha=0.7)
ax[0, 1].set_xticks(x_pos)
ax[0, 1].set_xticklabels(high_bits[:10], rotation=45)
ax[0, 1].legend()
ax[0, 1].set_title('Bits for High Delta_PCE')

ax[1, 0].hist(y, bins=40, color='steelblue', edgecolor='white')
ax[1, 0].axvline(median_y, color='red', linestyle='--', label=f'Median: {median_y:.3f}')
ax[1, 0].legend()
ax[1, 0].set_xlabel('Delta_PCE')
ax[1, 0].set_title('Delta_PCE Distribution')

models = results_df['Model'].tolist()
r2_vals = results_df['R2'].tolist()
r2_std = results_df['R2_std'].tolist()
ax[1, 1].bar(range(len(models)), r2_vals, yerr=r2_std, capsize=4, color=['steelblue', 'orange', 'green'][:len(models)])
ax[1, 1].set_xticks(range(len(models)))
ax[1, 1].set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
ax[1, 1].set_ylabel('R2')
ax[1, 1].set_title('Model Comparison')
ax[1, 1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{out_dir}/figures/krfp_analysis_summary.png', dpi=200)
plt.close()
print("    Saved: krfp_analysis_summary.png")

# Pattern analysis text
with open(f'{out_dir}/fingerprints/krfp_patterns.txt', 'w') as f:
    f.write("Klekota-Roth Fingerprints Pattern Analysis\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Samples: {len(fingerprints)}\n")
    f.write(f"Fingerprint: 4860 bits\n\n")
    f.write(f"Delta_PCE: mean={y.mean():.4f}, std={y.std():.4f}\n\n")
    f.write(f"PCA: PC1+PC2 = {sum(pca.explained_variance_ratio_[:2])*100:.2f}%\n\n")
    f.write("Top 50 Bits (MI):\n")
    for i in range(50):
        f.write(f"  {i+1:2d}. Bit {top50_idx[i]:4d} MI={top50_scores[i]:.6f}\n")
    f.write("\n\nBits for HIGH Delta_PCE:\n")
    for b in high_bits:
        f.write(f"  Bit {b:4d}: high={prev_high[b]:.3f} low={prev_low[b]:.3f} diff={diff[b]:.3f}\n")
    f.write("\n\nBits for LOW Delta_PCE:\n")
    for b in low_bits:
        f.write(f"  Bit {b:4d}: high={prev_high[b]:.3f} low={prev_low[b]:.3f} diff={diff[b]:.3f}\n")
    f.write("\n\nML Results:\n")
    for _, r in results_df.iterrows():
        f.write(f"  {r['Model']}: R2={r['R2']:.4f} (+/- {r['R2_std']:.4f})\n")
print("    Saved: krfp_patterns.txt")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nBest Model: {results_df.loc[results_df['R2'].idxmax(), 'Model']}")
print(f"R2 = {results_df['R2'].max():.4f}")
