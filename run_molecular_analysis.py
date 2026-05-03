#!/usr/bin/env python3
"""
Simplified Molecular Structure Analysis for QSPR
"""
import sys
import os

# Print progress to stdout
print("Starting molecular structure analysis...", flush=True)

import pandas as pd
import numpy as np
print("Pandas and numpy loaded", flush=True)

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
print("RDKit loaded", flush=True)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
print("Scikit-learn loaded", flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
print("Matplotlib loaded", flush=True)

import warnings
warnings.filterwarnings('ignore')

# File paths
INPUT_FILE = '/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx'
OUTPUT_DIR = '/share/yhm/test/AutoML_EDA'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'

os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80, flush=True)
print("MOLECULAR STRUCTURE ANALYSIS FOR QSPR", flush=True)
print("=" * 80, flush=True)

# Step 1: Load data
print("\n[Step 1] Loading data...", flush=True)
df = pd.read_excel(INPUT_FILE)
print(f"Loaded {len(df)} samples", flush=True)

# Calculate Delta_PCE
df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']

# Filter for valid data
cols_needed = ['smiles', 'Delta_PCE', 'molecular_weight', 'h_bond_donors',
               'h_bond_acceptors', 'rotatable_bonds', 'tpsa', 'log_p']
df_analysis = df[cols_needed].dropna(subset=['smiles', 'Delta_PCE'])
print(f"Samples with valid SMILES and Delta_PCE: {len(df_analysis)}", flush=True)

print(f"\nDelta_PCE statistics:", flush=True)
print(f"  Mean: {df_analysis['Delta_PCE'].mean():.4f}", flush=True)
print(f"  Std: {df_analysis['Delta_PCE'].std():.4f}", flush=True)
print(f"  Min: {df_analysis['Delta_PCE'].min():.4f}", flush=True)
print(f"  Max: {df_analysis['Delta_PCE'].max():.4f}", flush=True)

# Step 2: Calculate descriptors
print("\n[Step 2] Calculating molecular descriptors...", flush=True)

def calc_desc(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            'mol_weight': Descriptors.MolWt(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'num_rings': Descriptors.RingCount(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'tpsa_calc': Descriptors.TPSA(mol),
            'logp_calc': Descriptors.MolLogP(mol),
            'bertz_ct': Descriptors.BertzCT(mol),
            'num_hbd': Descriptors.NumHDonors(mol),
            'num_hba': Descriptors.NumHAcceptors(mol),
            'num_rot_bonds': Descriptors.NumRotatableBonds(mol),
            'num_hetero': Descriptors.NumHeteroatoms(mol),
            'fraction_csp3': Descriptors.FractionCSP3(mol),
        }
    except:
        return None

desc_list = []
valid_idx = []
for i, (idx, row) in enumerate(df_analysis.iterrows()):
    if i % 100 == 0:
        print(f"  Processing molecule {i}/{len(df_analysis)}", flush=True)
    d = calc_desc(row['smiles'])
    if d is not None:
        desc_list.append(d)
        valid_idx.append(idx)

print(f"Calculated descriptors for {len(desc_list)} molecules", flush=True)

df_desc = pd.DataFrame(desc_list, index=valid_idx)
df_combined = df_analysis.loc[valid_idx].copy()
for col in df_desc.columns:
    df_combined[col] = df_desc[col]

# Step 3: Generate fingerprints
print("\n[Step 3] Generating Morgan fingerprints...", flush=True)

def gen_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp)
    except:
        return None

fps = []
fp_idx = []
for i, (idx, row) in enumerate(df_combined.iterrows()):
    if i % 100 == 0:
        print(f"  Generating fingerprint {i}/{len(df_combined)}", flush=True)
    fp = gen_fp(row['smiles'])
    if fp is not None:
        fps.append(fp)
        fp_idx.append(idx)

fps = np.array(fps)
print(f"Generated fingerprints for {len(fps)} molecules", flush=True)

# Step 4: PCA
print("\n[Step 4] Performing PCA...", flush=True)

desc_cols = ['mol_weight', 'heavy_atoms', 'num_atoms', 'num_bonds', 'num_rings',
             'num_aromatic_rings', 'tpsa_calc', 'logp_calc', 'bertz_ct',
             'num_hbd', 'num_hba', 'num_rot_bonds', 'num_hetero', 'fraction_csp3']

X = df_combined.loc[fp_idx, desc_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA explained variance: {pca.explained_variance_ratio_}", flush=True)
print(f"Total: {sum(pca.explained_variance_ratio_):.2%}", flush=True)

# PCA plot
fig, ax = plt.subplots(figsize=(12, 10))
delta_vals = df_combined.loc[fp_idx, 'Delta_PCE'].values
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=delta_vals, cmap='RdYlBu_r',
                     s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(scatter, label='Delta_PCE')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
ax.set_title('PCA of Chemical Descriptors (Colored by Delta_PCE)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/pca_chemical_space.png', dpi=300, bbox_inches='tight')
print(f"Saved PCA plot to {FIGURES_DIR}/pca_chemical_space.png", flush=True)
plt.close()

# Step 5: t-SNE
print("\n[Step 5] Performing t-SNE...", flush=True)

perp = min(30, len(fps)-1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perp, n_iter=1000)
X_tsne = tsne.fit_transform(fps)

fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=delta_vals, cmap='RdYlBu_r',
                     s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(scatter, label='Delta_PCE')
ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title('t-SNE of Molecular Fingerprints (Colored by Delta_PCE)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/tsne_chemical_space.png', dpi=300, bbox_inches='tight')
print(f"Saved t-SNE plot to {FIGURES_DIR}/tsne_chemical_space.png", flush=True)
plt.close()

# Step 6: High/Low Delta_PCE analysis
print("\n[Step 6] Analyzing high/low Delta_PCE groups...", flush=True)

from scipy import stats

q75 = df_combined.loc[fp_idx, 'Delta_PCE'].quantile(0.75)
q25 = df_combined.loc[fp_idx, 'Delta_PCE'].quantile(0.25)

high_mask = df_combined.loc[fp_idx, 'Delta_PCE'] > q75
low_mask = df_combined.loc[fp_idx, 'Delta_PCE'] < q25

high_idx = df_combined.loc[fp_idx][high_mask].index
low_idx = df_combined.loc[fp_idx][low_mask].index

print(f"High group (top 25%): {len(high_idx)} molecules", flush=True)
print(f"Low group (bottom 25%): {len(low_idx)} molecules", flush=True)

# Statistical comparison
results = []
for col in desc_cols:
    high_vals = df_combined.loc[high_idx, col].dropna()
    low_vals = df_combined.loc[low_idx, col].dropna()
    if len(high_vals) > 0 and len(low_vals) > 0:
        stat, pval = stats.mannwhitneyu(high_vals, low_vals, alternative='two-sided')
        results.append({
            'Descriptor': col,
            'High_Mean': high_vals.mean(),
            'Low_Mean': low_vals.mean(),
            'Diff': high_vals.mean() - low_vals.mean(),
            'p_value': pval
        })

df_stats = pd.DataFrame(results).sort_values('p_value')
print("\nTop significant descriptors:", flush=True)
print(df_stats[df_stats['p_value'] < 0.05].to_string(), flush=True)

# Step 7: Save results
print("\n[Step 7] Saving results...", flush=True)

# Add coordinates to output
df_output = df_combined.copy()
df_output['PCA_1'] = np.nan
df_output['PCA_2'] = np.nan
df_output['tSNE_1'] = np.nan
df_output['tSNE_2'] = np.nan

for i, idx in enumerate(fp_idx):
    df_output.loc[idx, 'PCA_1'] = X_pca[i, 0]
    df_output.loc[idx, 'PCA_2'] = X_pca[i, 1]
    df_output.loc[idx, 'tSNE_1'] = X_tsne[i, 0]
    df_output.loc[idx, 'tSNE_2'] = X_tsne[i, 1]

df_output.to_csv(f'{OUTPUT_DIR}/extended_descriptors.csv', index=False)
print(f"Saved extended_descriptors.csv", flush=True)

# Summary
summary = f"""
================================================================================
MOLECULAR STRUCTURE ANALYSIS SUMMARY
================================================================================

Date: 2026-02-20

1. DATA OVERVIEW
----------------
Total samples: {len(df)}
Valid for analysis: {len(df_combined)}
With fingerprints: {len(fps)}

Delta_PCE Statistics:
  Mean: {df_combined['Delta_PCE'].mean():.4f}
  Std: {df_combined['Delta_PCE'].std():.4f}
  Min: {df_combined['Delta_PCE'].min():.4f}
  Max: {df_combined['Delta_PCE'].max():.4f}

2. PCA ANALYSIS
---------------
PC1 explains: {pca.explained_variance_ratio_[0]:.1%}
PC2 explains: {pca.explained_variance_ratio_[1]:.1%}
Total: {sum(pca.explained_variance_ratio_):.1%}

3. T-SNE ANALYSIS
-----------------
Performed on Morgan fingerprints (2048 bits, radius=2)
Perplexity: {perp}

4. GROUP COMPARISON
-------------------
High Delta_PCE (top 25%): {len(high_idx)} molecules
Low Delta_PCE (bottom 25%): {len(low_idx)} molecules

Significant descriptor differences (p < 0.05):
{df_stats[df_stats['p_value'] < 0.05].to_string() if len(df_stats[df_stats['p_value'] < 0.05]) > 0 else 'None found'}

5. OUTPUT FILES
---------------
Extended descriptors: {OUTPUT_DIR}/extended_descriptors.csv
PCA plot: {FIGURES_DIR}/pca_chemical_space.png
t-SNE plot: {FIGURES_DIR}/tsne_chemical_space.png
Analysis summary: {OUTPUT_DIR}/molecular_analysis.txt

================================================================================
"""

with open(f'{OUTPUT_DIR}/molecular_analysis.txt', 'w') as f:
    f.write(summary)

print(f"Saved molecular_analysis.txt", flush=True)

# Additional plot: descriptor distributions
if len(df_stats[df_stats['p_value'] < 0.05]) > 0:
    sig_cols = df_stats[df_stats['p_value'] < 0.05]['Descriptor'].tolist()[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(sig_cols):
        if i >= 6:
            break
        ax = axes[i]
        high_data = df_combined.loc[high_idx, col].dropna()
        low_data = df_combined.loc[low_idx, col].dropna()

        bp = ax.boxplot([high_data, low_data],
                        labels=['High Delta_PCE', 'Low Delta_PCE'],
                        patch_artist=True)
        colors = ['#ff9999', '#66b3ff']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        pval = df_stats[df_stats['Descriptor']==col]['p_value'].values[0]
        ax.set_title(f'{col}\np={pval:.4f}')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Descriptor Distributions: High vs Low Delta_PCE', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/descriptor_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved descriptor_distributions.png", flush=True)
    plt.close()

# Correlation plot
corr_cols = desc_cols + ['Delta_PCE']
corr = df_combined[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr[['Delta_PCE']].sort_values('Delta_PCE', ascending=False).drop('Delta_PCE'),
            annot=True, cmap='RdBu_r', center=0, ax=ax, fmt='.3f')
ax.set_title('Correlation with Delta_PCE', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/descriptor_correlations.png', dpi=300, bbox_inches='tight')
print(f"Saved descriptor_correlations.png", flush=True)
plt.close()

print("\n" + "=" * 80, flush=True)
print("ANALYSIS COMPLETE", flush=True)
print("=" * 80, flush=True)
