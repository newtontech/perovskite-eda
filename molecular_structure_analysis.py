#!/usr/bin/env python3
"""
Molecular Structure Analysis for QSPR
Analyzes SMILES data and Delta_PCE to identify structure-property relationships
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem import FragmentMatcher
from rdkit import DataStructs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# File paths
INPUT_FILE = '/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx'
OUTPUT_DIR = '/share/yhm/test/AutoML_EDA'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'

# Create output directories
import os
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("MOLECULAR STRUCTURE ANALYSIS FOR QSPR")
print("=" * 80)

# Step 1: Load data and calculate Delta_PCE
print("\n[Step 1] Loading data and calculating Delta_PCE...")
df = pd.read_excel(INPUT_FILE)
print(f"Loaded {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")

# Calculate Delta_PCE
df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']

# Filter for valid SMILES and Delta_PCE
df_analysis = df[['smiles', 'Delta_PCE', 'cas_number', 'pubchem_id',
                  'molecular_formula', 'molecular_weight', 'h_bond_donors',
                  'h_bond_acceptors', 'rotatable_bonds', 'tpsa', 'log_p']].dropna(subset=['smiles', 'Delta_PCE'])
print(f"Samples with valid SMILES and Delta_PCE: {len(df_analysis)}")

# Step 2: Calculate extended molecular descriptors using RDKit
print("\n[Step 2] Calculating extended molecular descriptors...")

def calculate_descriptors(smiles):
    """Calculate molecular descriptors from SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Calculate additional descriptors
        descriptors = {
            # Basic molecular properties
            'mol_weight_rdkit': Descriptors.MolWt(mol),
            'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'num_rings': Descriptors.RingCount(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_aliphatic_rings': Descriptors.NumAliphaticRings(mol),
            'num_saturated_rings': Descriptors.NumSaturatedRings(mol),

            # Electronic properties
            'tpsa_rdkit': Descriptors.TPSA(mol),
            'labute_asa': Descriptors.LabuteASA(mol),  # Approximate surface area
            'mol_logp_rdkit': Descriptors.MolLogP(mol),
            'mol_mr': Descriptors.MolMR(mol),  # Molecular refractivity

            # Complexity metrics
            'bertz_ct': Descriptors.BertzCT(mol),  # Bertz complexity
            'hall_kier_alpha': Descriptors.HallKierAlpha(mol),
            'kappa1': Descriptors.Kappa1(mol),  # Molecular shape indices
            'kappa2': Descriptors.Kappa2(mol),
            'kappa3': Descriptors.Kappa3(mol),

            # Hydrogen bonding
            'num_hbd': Descriptors.NumHDonors(mol),
            'num_hba': Descriptors.NumHAcceptors(mol),
            'num_hbd_lipinski': rdMolDescriptors.CalcNumLipinskiHBD(mol),
            'num_hba_lipinski': rdMolDescriptors.CalcNumLipinskiHBA(mol),

            # Rotatable bonds and flexibility
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_rotatable_bonds_rdkit': rdMolDescriptors.CalcNumRotatableBonds(mol),

            # Functional groups
            'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
            'num_valence_electrons': Descriptors.NumValenceElectrons(mol),
            'fraction_csp3': Descriptors.FractionCSP3(mol),  # Fraction of sp3 carbons

            # Aromaticity
            'num_aromatic_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()),
            'num_aromatic_carbons': Descriptors.NumAromaticCarbons(mol),
            'num_aromatic_nitrogens': Descriptors.NumAromaticNitrogens(mol),
            'num_aromatic_heterocycles': Descriptors.NumAromaticHeterocycles(mol),

            # Ring information
            'ring_count': rdMolDescriptors.CalcNumRings(mol),
            'aromatic_ratio': Descriptors.NumAromaticRings(mol) / max(Descriptors.RingCount(mol), 1),

            # Additional complexity
            'balaban_j': Descriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0,
            'morgan_similarity_mean': 0,  # Will be calculated later
        }

        # Calculate synthetic accessibility score approximation
        # Higher score = more complex/harder to synthesize
        descriptors['complexity_score'] = (
            descriptors['bertz_ct'] / 100 +
            descriptors['num_rings'] +
            descriptors['num_heteroatoms'] * 0.5 +
            descriptors['num_aromatic_rings'] * 0.5
        )

        return descriptors
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

# Calculate descriptors for all molecules
descriptor_list = []
valid_indices = []

for idx, row in df_analysis.iterrows():
    desc = calculate_descriptors(row['smiles'])
    if desc is not None:
        descriptor_list.append(desc)
        valid_indices.append(idx)

print(f"Successfully calculated descriptors for {len(descriptor_list)} molecules")

# Create descriptors dataframe
df_descriptors = pd.DataFrame(descriptor_list, index=valid_indices)

# Combine with original data
df_combined = df_analysis.loc[valid_indices].copy()
for col in df_descriptors.columns:
    df_combined[col] = df_descriptors[col]

print(f"Combined dataframe shape: {df_combined.shape}")

# Step 3: Generate Morgan fingerprints
print("\n[Step 3] Generating Morgan fingerprints...")

def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Generate Morgan fingerprint from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None

# Generate fingerprints
fingerprints = []
fp_valid_indices = []

for idx, row in df_combined.iterrows():
    fp = generate_morgan_fingerprint(row['smiles'])
    if fp is not None:
        fingerprints.append(fp)
        fp_valid_indices.append(idx)

fingerprints = np.array(fingerprints)
print(f"Generated fingerprints for {len(fingerprints)} molecules")
print(f"Fingerprint shape: {fingerprints.shape}")

# Step 4: Chemical Space Analysis with PCA
print("\n[Step 4] Performing PCA on chemical descriptors...")

# Select numerical descriptor columns for PCA
descriptor_cols = ['mol_weight_rdkit', 'heavy_atom_count', 'num_atoms', 'num_bonds',
                   'num_rings', 'num_aromatic_rings', 'tpsa_rdkit', 'mol_logp_rdkit',
                   'bertz_ct', 'num_hbd', 'num_hba', 'num_rotatable_bonds',
                   'num_heteroatoms', 'fraction_csp3', 'complexity_score']

X_descriptors = df_combined.loc[fp_valid_indices, descriptor_cols].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_descriptors)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Create PCA plot
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=df_combined.loc[fp_valid_indices, 'Delta_PCE'],
                     cmap='RdYlBu_r', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

cbar = plt.colorbar(scatter, label='Delta_PCE')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('PCA of Chemical Descriptors\nColored by Delta_PCE', fontsize=14)
ax.grid(True, alpha=0.3)

# Add zero line
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/pca_chemical_space.png', dpi=300, bbox_inches='tight')
print(f"Saved PCA plot to {FIGURES_DIR}/pca_chemical_space.png")
plt.close()

# Step 5: t-SNE Visualization
print("\n[Step 5] Performing t-SNE visualization...")

# Use fingerprints for t-SNE (more suitable for structural similarity)
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fingerprints)-1),
            n_iter=1000, learning_rate='auto', init='pca')
X_tsne = tsne.fit_transform(fingerprints)

# Create t-SNE plot
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                     c=df_combined.loc[fp_valid_indices, 'Delta_PCE'],
                     cmap='RdYlBu_r', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

cbar = plt.colorbar(scatter, label='Delta_PCE')
ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title('t-SNE of Molecular Fingerprints\nColored by Delta_PCE', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/tsne_chemical_space.png', dpi=300, bbox_inches='tight')
print(f"Saved t-SNE plot to {FIGURES_DIR}/tsne_chemical_space.png")
plt.close()

# Step 6: Identify structural motifs for high/low Delta_PCE
print("\n[Step 6] Analyzing structural motifs for high/low Delta_PCE...")

# Define high and low Delta_PCE groups
delta_pce_values = df_combined.loc[fp_valid_indices, 'Delta_PCE']
median_delta = delta_pce_values.median()

high_group_mask = delta_pce_values > delta_pce_values.quantile(0.75)
low_group_mask = delta_pce_values < delta_pce_values.quantile(0.25)

high_group_idx = df_combined.loc[fp_valid_indices][high_group_mask].index
low_group_idx = df_combined.loc[fp_valid_indices][low_group_mask].index

print(f"High Delta_PCE group (top 25%): {len(high_group_idx)} molecules")
print(f"Low Delta_PCE group (bottom 25%): {len(low_group_idx)} molecules")

# Analyze descriptor differences between groups
high_descriptors = df_combined.loc[high_group_idx, descriptor_cols]
low_descriptors = df_combined.loc[low_group_idx, descriptor_cols]

# Statistical comparison
from scipy import stats

statistical_results = []
for col in descriptor_cols:
    if col in high_descriptors.columns and col in low_descriptors.columns:
        stat, pval = stats.mannwhitneyu(high_descriptors[col].dropna(),
                                         low_descriptors[col].dropna(),
                                         alternative='two-sided')
        high_mean = high_descriptors[col].mean()
        low_mean = low_descriptors[col].mean()
        statistical_results.append({
            'Descriptor': col,
            'High_Mean': high_mean,
            'Low_Mean': low_mean,
            'Difference': high_mean - low_mean,
            'p_value': pval,
            'Significant': pval < 0.05
        })

df_stats = pd.DataFrame(statistical_results)
df_stats = df_stats.sort_values('p_value')

print("\nSignificant descriptors (p < 0.05):")
print(df_stats[df_stats['Significant']][['Descriptor', 'High_Mean', 'Low_Mean', 'Difference', 'p_value']])

# Step 7: Substructure Analysis
print("\n[Step 7] Substructure analysis...")

# Define common functional groups/moieties to look for
substructure_smarts = {
    'Aromatic ring': '[a]',
    'Benzene': 'c1ccccc1',
    'Pyridine': 'c1ccncc1',
    'Carbonyl': '[C]=[O]',
    'Ester': '[C](=[O])[O]',
    'Amide': '[C](=[O])[N]',
    'Amine': '[N;!+]',
    'Primary amine': '[NH2]',
    'Secondary amine': '[NH]([!H])[!H]',
    'Tertiary amine': '[N]([!H])([!H])[!H]',
    'Alcohol': '[OH]',
    'Carboxylic acid': '[C](=[O])[OH]',
    'Halogen': '[F,Cl,Br,I]',
    'Sulfur': '[S]',
    'Phosphorus': '[P]',
    'Nitro': '[N+](=[O])[O-]',
    'Nitrile': '[C]#[N]',
    'Ether': '[O]([C])[C]',
    'Ketone': '[C](=[O])[C]',
    'Aldehyde': '[CH]=[O]',
    'Thiol': '[SH]',
    'Double bond': '[C]=[C]',
    'Triple bond': '[C]#[C]',
}

def count_substructure(smiles, smarts):
    """Count occurrences of a substructure in a molecule"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        pattern = Chem.MolFromSmarts(smarts)
        if mol is None or pattern is None:
            return 0
        return len(mol.GetSubstructMatches(pattern))
    except:
        return 0

# Analyze substructures
substructure_results = []

for name, smarts in substructure_smarts.items():
    high_counts = []
    low_counts = []

    for idx in high_group_idx:
        count = count_substructure(df_combined.loc[idx, 'smiles'], smarts)
        high_counts.append(count)

    for idx in low_group_idx:
        count = count_substructure(df_combined.loc[idx, 'smiles'], smarts)
        low_counts.append(count)

    # Calculate frequency (proportion of molecules containing the group)
    high_freq = sum(1 for c in high_counts if c > 0) / len(high_counts) if high_counts else 0
    low_freq = sum(1 for c in low_counts if c > 0) / len(low_counts) if low_counts else 0

    # Statistical test
    if sum(high_counts) > 0 or sum(low_counts) > 0:
        stat, pval = stats.mannwhitneyu(high_counts, low_counts, alternative='two-sided')
    else:
        pval = 1.0

    substructure_results.append({
        'Substructure': name,
        'SMARTS': smarts,
        'High_Frequency': high_freq,
        'Low_Frequency': low_freq,
        'Frequency_Diff': high_freq - low_freq,
        'High_Mean_Count': np.mean(high_counts) if high_counts else 0,
        'Low_Mean_Count': np.mean(low_counts) if low_counts else 0,
        'p_value': pval,
        'Significant': pval < 0.05
    })

df_substructure = pd.DataFrame(substructure_results)
df_substructure = df_substructure.sort_values('p_value')

print("\nSignificant substructures (p < 0.05):")
print(df_substructure[df_substructure['Significant']][['Substructure', 'High_Frequency', 'Low_Frequency', 'Frequency_Diff', 'p_value']])

# Step 8: Save extended descriptors
print("\n[Step 8] Saving extended descriptors...")

# Prepare output dataframe
df_output = df_combined.copy()
df_output['PCA_1'] = np.nan
df_output['PCA_2'] = np.nan
df_output['tSNE_1'] = np.nan
df_output['tSNE_2'] = np.nan

# Add PCA and t-SNE coordinates
for i, idx in enumerate(fp_valid_indices):
    df_output.loc[idx, 'PCA_1'] = X_pca[i, 0]
    df_output.loc[idx, 'PCA_2'] = X_pca[i, 1]
    df_output.loc[idx, 'tSNE_1'] = X_tsne[i, 0]
    df_output.loc[idx, 'tSNE_2'] = X_tsne[i, 1]

# Select columns to save
output_cols = ['smiles', 'Delta_PCE', 'cas_number', 'pubchem_id', 'molecular_formula'] + \
              [col for col in df_output.columns if col not in
               ['smiles', 'Delta_PCE', 'cas_number', 'pubchem_id', 'molecular_formula'] and
               col in df_output.columns]

df_output[output_cols].to_csv(f'{OUTPUT_DIR}/extended_descriptors.csv', index=False)
print(f"Saved extended descriptors to {OUTPUT_DIR}/extended_descriptors.csv")

# Step 9: Generate analysis summary
print("\n[Step 9] Generating analysis summary...")

summary = f"""
================================================================================
MOLECULAR STRUCTURE ANALYSIS SUMMARY FOR QSPR
================================================================================

Date: 2026-02-20
Input file: {INPUT_FILE}

1. DATA OVERVIEW
----------------
Total samples in dataset: {len(df)}
Samples with valid SMILES and Delta_PCE: {len(df_analysis)}
Samples with calculated descriptors: {len(df_combined)}
Samples with fingerprints: {len(fingerprints)}

Delta_PCE Statistics:
  Mean: {delta_pce_values.mean():.4f}
  Median: {median_delta:.4f}
  Std Dev: {delta_pce_values.std():.4f}
  Min: {delta_pce_values.min():.4f}
  Max: {delta_pce_values.max():.4f}

2. MOLECULAR DESCRIPTORS
------------------------
Total descriptors calculated: {len(descriptor_cols)}
Descriptors include:
  - Basic molecular properties (weight, atom/bond counts, rings)
  - Electronic properties (TPSA, LogP, molecular refractivity)
  - Complexity metrics (Bertz complexity, shape indices)
  - Hydrogen bonding (donors/acceptors)
  - Rotatable bonds and flexibility
  - Functional groups and aromaticity

3. PCA ANALYSIS
---------------
Principal Component 1 explains: {pca.explained_variance_ratio_[0]:.1%} of variance
Principal Component 2 explains: {pca.explained_variance_ratio_[1]:.1%} of variance
Total variance explained: {sum(pca.explained_variance_ratio_):.1%}

Top loadings for PC1:
"""

# Add PCA loadings
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=descriptor_cols)
loadings_sorted = loadings['PC1'].abs().sort_values(ascending=False).head(5)
for feat in loadings_sorted.index:
    summary += f"  {feat}: {loadings.loc[feat, 'PC1']:.3f}\n"

summary += f"""
4. T-SNE ANALYSIS
-----------------
t-SNE performed on Morgan fingerprints (2048 bits, radius=2)
Perplexity: {min(30, len(fingerprints)-1)}
Visualization shows clustering of molecules by structural similarity

5. STRUCTURAL MOTIFS ANALYSIS
-----------------------------
High Delta_PCE group (top 25%): {len(high_group_idx)} molecules
  Mean Delta_PCE: {df_combined.loc[high_group_idx, 'Delta_PCE'].mean():.4f}

Low Delta_PCE group (bottom 25%): {len(low_group_idx)} molecules
  Mean Delta_PCE: {df_combined.loc[low_group_idx, 'Delta_PCE'].mean():.4f}

SIGNIFICANT DESCRIPTOR DIFFERENCES (p < 0.05):
"""

if len(df_stats[df_stats['Significant']]) > 0:
    for _, row in df_stats[df_stats['Significant']].iterrows():
        direction = "HIGHER" if row['Difference'] > 0 else "LOWER"
        summary += f"  - {row['Descriptor']}: {direction} in high group "
        summary += f"(diff={row['Difference']:.3f}, p={row['p_value']:.4f})\n"
else:
    summary += "  No statistically significant differences found\n"

summary += f"""
SIGNIFICANT SUBSTRUCTURE DIFFERENCES (p < 0.05):
"""

if len(df_substructure[df_substructure['Significant']]) > 0:
    for _, row in df_substructure[df_substructure['Significant']].iterrows():
        direction = "MORE" if row['Frequency_Diff'] > 0 else "LESS"
        summary += f"  - {row['Substructure']}: {direction} in high group "
        summary += f"({row['High_Frequency']:.1%} vs {row['Low_Frequency']:.1%}, p={row['p_value']:.4f})\n"
else:
    summary += "  No statistically significant substructure differences found\n"

summary += f"""
6. OUTPUT FILES
---------------
Extended descriptors: {OUTPUT_DIR}/extended_descriptors.csv
PCA plot: {FIGURES_DIR}/pca_chemical_space.png
t-SNE plot: {FIGURES_DIR}/tsne_chemical_space.png
Analysis summary: {OUTPUT_DIR}/molecular_analysis.txt

7. KEY FINDINGS
---------------
"""

# Add key findings based on analysis
if len(df_stats[df_stats['Significant']]) > 0:
    summary += "\nDescriptors associated with higher Delta_PCE:\n"
    higher_in_high = df_stats[df_stats['Significant'] & (df_stats['Difference'] > 0)]
    for _, row in higher_in_high.head(5).iterrows():
        summary += f"  - {row['Descriptor']}\n"

    summary += "\nDescriptors associated with lower Delta_PCE:\n"
    higher_in_low = df_stats[df_stats['Significant'] & (df_stats['Difference'] < 0)]
    for _, row in higher_in_low.head(5).iterrows():
        summary += f"  - {row['Descriptor']}\n"

summary += """
8. RECOMMENDATIONS FOR QSPR MODELING
------------------------------------
1. Use Morgan fingerprints for structural similarity-based models
2. Include Bertz complexity and TPSA as important descriptors
3. Consider ring count and aromatic features in feature selection
4. Apply the extended descriptor set for interpretable models
5. Use PCA/t-SNE coordinates for dimensionality reduction if needed

================================================================================
"""

# Save summary
with open(f'{OUTPUT_DIR}/molecular_analysis.txt', 'w') as f:
    f.write(summary)

print(f"Saved analysis summary to {OUTPUT_DIR}/molecular_analysis.txt")

# Step 10: Create additional visualization - Descriptor distributions
print("\n[Step 10] Creating additional visualizations...")

# Box plots for significant descriptors
if len(df_stats[df_stats['Significant']]) > 0:
    sig_descriptors = df_stats[df_stats['Significant']]['Descriptor'].tolist()[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, desc in enumerate(sig_descriptors):
        if i >= 6:
            break
        ax = axes[i]

        data_high = df_combined.loc[high_group_idx, desc].dropna()
        data_low = df_combined.loc[low_group_idx, desc].dropna()

        bp = ax.boxplot([data_high, data_low], labels=['High Delta_PCE', 'Low Delta_PCE'],
                        patch_artist=True)

        colors = ['#ff9999', '#66b3ff']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(f'{desc}\np={df_stats[df_stats["Descriptor"]==desc]["p_value"].values[0]:.4f}')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Descriptor Distributions: High vs Low Delta_PCE Groups', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/descriptor_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved descriptor distributions plot to {FIGURES_DIR}/descriptor_distributions.png")
    plt.close()

# Correlation heatmap of descriptors with Delta_PCE
numeric_cols = descriptor_cols + ['Delta_PCE']
corr_matrix = df_combined[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix[['Delta_PCE']].sort_values('Delta_PCE', ascending=False).drop('Delta_PCE'),
            annot=True, cmap='RdBu_r', center=0, ax=ax, fmt='.3f')
ax.set_title('Correlation of Descriptors with Delta_PCE', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/descriptor_correlations.png', dpi=300, bbox_inches='tight')
print(f"Saved correlation plot to {FIGURES_DIR}/descriptor_correlations.png")
plt.close()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nAll results saved to: {OUTPUT_DIR}")
