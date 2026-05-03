#!/usr/bin/env python3
"""
Quick Molecular Structure Analysis for QSPR - Memory Optimized
"""
import sys
import gc
import os

print("=== MOLECULAR ANALYSIS STARTING ===", flush=True)

# Load libraries one by one
print("Loading libraries...", flush=True)
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded successfully", flush=True)

# Configuration
INPUT_FILE = '/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx'
OUTPUT_DIR = '/share/yhm/test/AutoML_EDA'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'

os.makedirs(FIGURES_DIR, exist_ok=True)

# Step 1: Load data
print("\n[1/7] Loading data...", flush=True)
df = pd.read_excel(INPUT_FILE, usecols=['smiles', 'jv_reverse_scan_pce', 'jv_reverse_scan_pce_without_modulator'])
print(f"Loaded {len(df)} rows", flush=True)

# Calculate Delta_PCE
df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']

# Drop rows with missing values
df = df.dropna(subset=['smiles', 'Delta_PCE'])
df = df.reset_index(drop=True)
print(f"After filtering: {len(df)} samples", flush=True)
print(f"Delta_PCE - Mean: {df['Delta_PCE'].mean():.4f}, Std: {df['Delta_PCE'].std():.4f}", flush=True)

# Step 2: Calculate descriptors
print("\n[2/7] Calculating descriptors...", flush=True)

def calc_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            'mw': Descriptors.MolWt(mol),
            'heavy': Descriptors.HeavyAtomCount(mol),
            'rings': Descriptors.RingCount(mol),
            'aromatic': Descriptors.NumAromaticRings(mol),
            'tpsa': Descriptors.TPSA(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rot': Descriptors.NumRotatableBonds(mol),
            'hetero': Descriptors.NumHeteroatoms(mol),
            'fsp3': Descriptors.FractionCSP3(mol),
            'bertz': Descriptors.BertzCT(mol),
        }
    except:
        return None

# Process in batches
batch_size = 500
all_desc = []
valid_indices = []

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    for idx, row in batch.iterrows():
        d = calc_descriptors(row['smiles'])
        if d is not None:
            all_desc.append(d)
            valid_indices.append(idx)
    print(f"  Processed {min(i+batch_size, len(df))}/{len(df)}", flush=True)

df_desc = pd.DataFrame(all_desc)
df_valid = df.iloc[valid_indices].reset_index(drop=True)
df_combined = pd.concat([df_valid, df_desc], axis=1)
print(f"Valid molecules with descriptors: {len(df_combined)}", flush=True)

del df, df_desc, df_valid, all_desc
gc.collect()

# Step 3: Generate fingerprints
print("\n[3/7] Generating fingerprints...", flush=True)

def gen_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp)
    except:
        return None

fingerprints = []
fp_indices = []

for i, (idx, row) in enumerate(df_combined.iterrows()):
    fp = gen_fingerprint(row['smiles'])
    if fp is not None:
        fingerprints.append(fp)
        fp_indices.append(idx)
    if (i+1) % 200 == 0:
        print(f"  Generated {i+1}/{len(df_combined)}", flush=True)

fingerprints = np.array(fingerprints)
df_fp = df_combined.iloc[fp_indices].reset_index(drop=True)
print(f"Fingerprints generated: {len(fingerprints)}", flush=True)

del df_combined
gc.collect()

# Step 4: PCA
print("\n[4/7] Performing PCA...", flush=True)

desc_cols = ['mw', 'heavy', 'rings', 'aromatic', 'tpsa', 'logp', 'hbd', 'hba', 'rot', 'hetero', 'fsp3', 'bertz']
X = df_fp[desc_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}", flush=True)

# Save PCA plot
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=df_fp['Delta_PCE'].values, cmap='RdYlBu_r',
                     s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(scatter, label='Delta_PCE')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
ax.set_title('PCA of Chemical Descriptors (Colored by Delta_PCE)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/pca_chemical_space.png', dpi=300, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR}/pca_chemical_space.png", flush=True)
plt.close()

# Step 5: t-SNE
print("\n[5/7] Performing t-SNE...", flush=True)

perp = min(30, len(fingerprints)-1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perp, n_iter=500, learning_rate='auto', init='pca')
X_tsne = tsne.fit_transform(fingerprints)

fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                     c=df_fp['Delta_PCE'].values, cmap='RdYlBu_r',
                     s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(scatter, label='Delta_PCE')
ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title('t-SNE of Molecular Fingerprints (Colored by Delta_PCE)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/tsne_chemical_space.png', dpi=300, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR}/tsne_chemical_space.png", flush=True)
plt.close()

# Step 6: Statistical analysis
print("\n[6/7] Analyzing high/low Delta_PCE groups...", flush=True)

q75 = df_fp['Delta_PCE'].quantile(0.75)
q25 = df_fp['Delta_PCE'].quantile(0.25)

high_mask = df_fp['Delta_PCE'] > q75
low_mask = df_fp['Delta_PCE'] < q25

high_group = df_fp[high_mask]
low_group = df_fp[low_mask]

print(f"High group (Q4): {len(high_group)}, Low group (Q1): {len(low_group)}", flush=True)

# Compare descriptors
stat_results = []
for col in desc_cols:
    stat, pval = stats.mannwhitneyu(high_group[col].dropna(), low_group[col].dropna(), alternative='two-sided')
    stat_results.append({
        'Descriptor': col,
        'High_Mean': high_group[col].mean(),
        'Low_Mean': low_group[col].mean(),
        'Difference': high_group[col].mean() - low_group[col].mean(),
        'p_value': pval
    })

df_stats = pd.DataFrame(stat_results).sort_values('p_value')
sig_stats = df_stats[df_stats['p_value'] < 0.05]
print(f"\nSignificant descriptors (p<0.05): {len(sig_stats)}", flush=True)
if len(sig_stats) > 0:
    print(sig_stats.to_string(), flush=True)

# Step 7: Save all outputs
print("\n[7/7] Saving outputs...", flush=True)

# Save extended descriptors
df_output = df_fp.copy()
df_output['PCA_1'] = X_pca[:, 0]
df_output['PCA_2'] = X_pca[:, 1]
df_output['tSNE_1'] = X_tsne[:, 0]
df_output['tSNE_2'] = X_tsne[:, 1]
df_output.to_csv(f'{OUTPUT_DIR}/extended_descriptors.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/extended_descriptors.csv", flush=True)

# Save descriptor distributions plot
if len(sig_stats) > 0:
    sig_cols = sig_stats['Descriptor'].tolist()[:6]
    n_plots = min(6, len(sig_cols))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(sig_cols):
        if i >= 6:
            break
        ax = axes[i]
        bp = ax.boxplot([high_group[col].dropna(), low_group[col].dropna()],
                        labels=['High\nDelta_PCE', 'Low\nDelta_PCE'],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#ff9999', '#66b3ff']):
            patch.set_facecolor(color)
        pval = sig_stats[sig_stats['Descriptor']==col]['p_value'].values[0]
        ax.set_title(f'{col}\np={pval:.4f}', fontsize=11)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_plots, 6):
        axes[i].set_visible(False)

    plt.suptitle('Descriptor Distributions: High vs Low Delta_PCE Groups', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/descriptor_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR}/descriptor_distributions.png", flush=True)
    plt.close()

# Save correlation plot
corr_cols = desc_cols + ['Delta_PCE']
corr = df_fp[corr_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr[['Delta_PCE']].sort_values('Delta_PCE', ascending=False).drop('Delta_PCE'),
            annot=True, cmap='RdBu_r', center=0, ax=ax, fmt='.3f', cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation of Descriptors with Delta_PCE', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/descriptor_correlations.png', dpi=300, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR}/descriptor_correlations.png", flush=True)
plt.close()

# Generate summary report
summary = f"""
================================================================================
MOLECULAR STRUCTURE ANALYSIS SUMMARY FOR QSPR
================================================================================

Analysis Date: 2026-02-20
Input File: {INPUT_FILE}

1. DATA OVERVIEW
----------------
Total samples analyzed: {len(df_fp)}
Descriptor features calculated: {len(desc_cols)}

Delta_PCE Statistics:
  Mean: {df_fp['Delta_PCE'].mean():.4f}
  Std Dev: {df_fp['Delta_PCE'].std():.4f}
  Min: {df_fp['Delta_PCE'].min():.4f}
  Max: {df_fp['Delta_PCE'].max():.4f}
  Median: {df_fp['Delta_PCE'].median():.4f}

2. MOLECULAR DESCRIPTORS CALCULATED
-----------------------------------
- Molecular weight (mw)
- Heavy atom count (heavy)
- Ring count (rings)
- Aromatic ring count (aromatic)
- Topological polar surface area (tpsa)
- LogP (logp)
- Hydrogen bond donors (hbd)
- Hydrogen bond acceptors (hba)
- Rotatable bonds (rot)
- Heteroatom count (hetero)
- Fraction sp3 carbons (fsp3)
- Bertz complexity (bertz)

3. PCA ANALYSIS
---------------
Principal Component 1 (PC1): {pca.explained_variance_ratio_[0]:.1%} variance
Principal Component 2 (PC2): {pca.explained_variance_ratio_[1]:.1%} variance
Total variance explained: {sum(pca.explained_variance_ratio_):.1%}

4. T-SNE ANALYSIS
-----------------
Morgan fingerprints: 2048 bits, radius=2
Perplexity: {perp}
Iterations: 500

5. GROUP COMPARISON (HIGH vs LOW Delta_PCE)
-------------------------------------------
High Delta_PCE group (Q4, >{q75:.4f}): {len(high_group)} molecules
Low Delta_PCE group (Q1, <{q25:.4f}): {len(low_group)} molecules

SIGNIFICANT DESCRIPTOR DIFFERENCES (p < 0.05):
"""

if len(sig_stats) > 0:
    for _, row in sig_stats.iterrows():
        direction = "HIGHER" if row['Difference'] > 0 else "LOWER"
        summary += f"  - {row['Descriptor']}: {direction} in high group "
        summary += f"(diff={row['Difference']:.3f}, p={row['p_value']:.4f})\n"
else:
    summary += "  No statistically significant differences found (p >= 0.05)\n"

summary += f"""
6. CORRELATIONS WITH Delta_PCE
------------------------------
"""

corr_with_pce = corr['Delta_PCE'].drop('Delta_PCE').sort_values(key=abs, ascending=False)
for desc, r in corr_with_pce.items():
    summary += f"  {desc}: r = {r:.3f}\n"

summary += f"""
7. OUTPUT FILES
---------------
- Extended descriptors: {OUTPUT_DIR}/extended_descriptors.csv
- PCA plot: {FIGURES_DIR}/pca_chemical_space.png
- t-SNE plot: {FIGURES_DIR}/tsne_chemical_space.png
- Descriptor distributions: {FIGURES_DIR}/descriptor_distributions.png
- Correlation heatmap: {FIGURES_DIR}/descriptor_correlations.png
- This summary: {OUTPUT_DIR}/molecular_analysis.txt

8. KEY FINDINGS AND RECOMMENDATIONS
------------------------------------
"""

if len(sig_stats) > 0:
    higher_in_high = sig_stats[sig_stats['Difference'] > 0]['Descriptor'].tolist()
    higher_in_low = sig_stats[sig_stats['Difference'] < 0]['Descriptor'].tolist()

    if higher_in_high:
        summary += "\nDescriptors ASSOCIATED WITH HIGHER Delta_PCE:\n"
        for d in higher_in_high:
            summary += f"  - {d}\n"

    if higher_in_low:
        summary += "\nDescriptors ASSOCIATED WITH LOWER Delta_PCE:\n"
        for d in higher_in_low:
            summary += f"  - {d}\n"

summary += """
For QSPR modeling:
1. Use Morgan fingerprints for structure-activity models (e.g., Random Forest, XGBoost)
2. Include significant descriptors as features for interpretable models
3. Consider PCA for dimensionality reduction if needed
4. Use t-SNE visualization to identify chemical space clusters

================================================================================
END OF REPORT
================================================================================
"""

with open(f'{OUTPUT_DIR}/molecular_analysis.txt', 'w') as f:
    f.write(summary)
print(f"Saved: {OUTPUT_DIR}/molecular_analysis.txt", flush=True)

print("\n" + "=" * 60, flush=True)
print("ANALYSIS COMPLETE!", flush=True)
print("=" * 60, flush=True)
print(f"\nAll results saved to: {OUTPUT_DIR}", flush=True)
print("Generated files:", flush=True)
print(f"  - extended_descriptors.csv", flush=True)
print(f"  - molecular_analysis.txt", flush=True)
print(f"  - figures/pca_chemical_space.png", flush=True)
print(f"  - figures/tsne_chemical_space.png", flush=True)
print(f"  - figures/descriptor_distributions.png", flush=True)
print(f"  - figures/descriptor_correlations.png", flush=True)
