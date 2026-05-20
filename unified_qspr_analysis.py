#!/usr/bin/env python3
"""
Unified QSPR Analysis for Perovskite Solar Cell Modulators
This script performs comprehensive analysis including:
- Data preprocessing
- Extended RDKit descriptors
- Molecular fingerprints (ECFP, MACCS, KRFP)
- ML modeling
- Statistical analysis
- Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directories
for d in ['figures', 'tables', 'fingerprints', 'report']:
    os.makedirs(f'/share/yhm/test/AutoML_EDA/{d}', exist_ok=True)

# File paths
INPUT_FILE = "/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx"
OUTPUT_DIR = "/share/yhm/test/AutoML_EDA"

print("=" * 70)
print("UNIFIED QSPR ANALYSIS FOR PEROVSKITE SOLAR CELL MODULATORS")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# STEP 1: Load and Preprocess Data
# ============================================================================
print("\n[Step 1] Loading and preprocessing data...")
df = pd.read_excel(INPUT_FILE)
initial_samples = len(df)
print(f"Loaded {initial_samples} samples with {len(df.columns)} columns")

# Convert PCE columns to numeric
pce_cols = ['jv_reverse_scan_pce', 'jv_reverse_scan_pce_without_modulator']
for col in pce_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate Delta_PCE
df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']

# Select relevant columns
chemical_features = ['molecular_weight', 'h_bond_donors', 'h_bond_acceptors',
                     'rotatable_bonds', 'tpsa', 'log_p']
identifier_cols = ['cas_number', 'pubchem_id', 'smiles', 'molecular_formula']
other_cols = ['jv_reverse_scan_pce', 'jv_reverse_scan_pce_without_modulator',
              'jv_reverse_scan_j_sc', 'jv_reverse_scan_v_oc', 'jv_reverse_scan_ff']

all_cols = identifier_cols + chemical_features + other_cols + ['Delta_PCE']
selected_cols = [c for c in all_cols if c in df.columns]
df_selected = df[selected_cols].copy()

# Remove rows with missing values in key columns
analysis_cols = chemical_features + ['Delta_PCE']
df_clean = df_selected.dropna(subset=analysis_cols).copy()
final_samples = len(df_clean)

print(f"After cleaning: {final_samples} samples ({100*final_samples/initial_samples:.2f}% retained)")

# Save processed data
df_clean.to_csv(f'{OUTPUT_DIR}/processed_data.csv', index=False)
print(f"Saved processed data to {OUTPUT_DIR}/processed_data.csv")

# ============================================================================
# STEP 2: Basic Statistics and Correlations
# ============================================================================
print("\n[Step 2] Computing basic statistics...")

# Chemical features statistics
chem_stats = df_clean[chemical_features].describe()
chem_stats.to_csv(f'{OUTPUT_DIR}/tables/chemical_features_stats.csv')

# Delta_PCE statistics
pce_stats = df_clean['Delta_PCE'].describe()
print("\nDelta_PCE Statistics:")
print(pce_stats)
print(f"\nSkewness: {df_clean['Delta_PCE'].skew():.4f}")
print(f"Kurtosis: {df_clean['Delta_PCE'].kurtosis():.4f}")
print(f"Positive: {(df_clean['Delta_PCE'] > 0).sum()}, Negative: {(df_clean['Delta_PCE'] < 0).sum()}")

# Correlation analysis
print("\n[Step 3] Computing correlations...")
correlations = df_clean[chemical_features + ['Delta_PCE']].corr()
correlations.to_csv(f'{OUTPUT_DIR}/tables/correlation_matrix.csv')

# Correlation with Delta_PCE
corr_with_pce = correlations['Delta_PCE'].drop('Delta_PCE').sort_values(key=abs, ascending=False)
print("\nCorrelations with Delta_PCE:")
for feat, corr in corr_with_pce.items():
    print(f"  {feat}: {corr:.4f}")

# ============================================================================
# STEP 3: Visualizations
# ============================================================================
print("\n[Step 4] Creating visualizations...")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Figure 1: Delta_PCE Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1 = axes[0]
ax1.hist(df_clean['Delta_PCE'], bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax1.axvline(x=df_clean['Delta_PCE'].mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {df_clean["Delta_PCE"].mean():.2f}')
ax1.set_xlabel('Delta_PCE (%)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Delta_PCE', fontsize=14)
ax1.legend()

# Box plot by sign
ax2 = axes[1]
df_clean['PCE_Category'] = pd.cut(df_clean['Delta_PCE'],
                                   bins=[-np.inf, -0.1, 0.1, np.inf],
                                   labels=['Negative', 'Near Zero', 'Positive'])
df_clean.boxplot(column='Delta_PCE', by='PCE_Category', ax=ax2)
ax2.set_xlabel('PCE Change Category', fontsize=12)
ax2.set_ylabel('Delta_PCE (%)', fontsize=12)
ax2.set_title('Delta_PCE by Category', fontsize=14)
plt.suptitle('')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig1_delta_pce_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig1_delta_pce_distribution.png")

# Figure 2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(correlations, dtype=bool))
sns.heatmap(correlations, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            mask=mask, square=True, ax=ax)
ax.set_title('Correlation Matrix of Chemical Features and Delta_PCE', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig2_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig2_correlation_heatmap.png")

# Figure 3: Scatter plots of top correlated features
top_features = corr_with_pce.head(4).index.tolist()
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, feat in enumerate(top_features):
    ax = axes[i//2, i%2]
    ax.scatter(df_clean[feat], df_clean['Delta_PCE'], alpha=0.3, s=10)
    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel('Delta_PCE', fontsize=11)
    ax.set_title(f'{feat} vs Delta_PCE (r={corr_with_pce[feat]:.3f})', fontsize=12)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig3_feature_scatter_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_feature_scatter_plots.png")

# Figure 4: Feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, feat in enumerate(chemical_features):
    ax = axes[i//3, i%3]
    ax.hist(df_clean[feat], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Distribution of {feat}', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig4_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_feature_distributions.png")

# ============================================================================
# STEP 4: RDKit Extended Descriptors (if available)
# ============================================================================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.QED import qed
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    print("\n[Step 5] Calculating extended RDKit descriptors...")

    def calculate_extended_descriptors(smiles):
        """Calculate extended RDKit descriptors from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        desc = {}
        # Atom counts
        desc['C'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        desc['H'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'H')
        desc['N'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
        desc['O'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
        desc['F'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')

        # RDKit descriptors
        desc['MW_rdkit'] = Descriptors.MolWt(mol)
        desc['LogP_rdkit'] = Descriptors.MolLogP(mol)
        desc['TPSA_rdkit'] = Descriptors.TPSA(mol)
        desc['H_acceptor_rdkit'] = Lipinski.NumHAcceptors(mol)
        desc['H_donor_rdkit'] = Lipinski.NumHDonors(mol)
        desc['RB_rdkit'] = Lipinski.NumRotatableBonds(mol)
        desc['Aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        desc['Aliphatic_rings'] = rdMolDescriptors.CalcNumAliphaticRings(mol)
        desc['Saturated_rings'] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        desc['Heteroatoms'] = rdMolDescriptors.CalcNumHeteroatoms(mol)
        desc['QED'] = qed(mol)
        desc['Heavy_atoms'] = mol.GetNumHeavyAtoms()
        desc['Ring_count'] = rdMolDescriptors.CalcNumRings(mol)

        return desc

    # Calculate for all molecules
    rdkit_data = []
    valid_indices = []
    for i, smiles in enumerate(df_clean['smiles'].values):
        if pd.notna(smiles):
            desc = calculate_extended_descriptors(str(smiles))
            if desc:
                rdkit_data.append(desc)
                valid_indices.append(i)

    if rdkit_data:
        rdkit_df = pd.DataFrame(rdkit_data, index=valid_indices)
        rdkit_df.to_csv(f'{OUTPUT_DIR}/fingerprints/rdkit_extended_descriptors.csv')

        # Merge with Delta_PCE for analysis
        rdkit_df['Delta_PCE'] = df_clean.loc[valid_indices, 'Delta_PCE'].values

        # Correlation analysis
        rdkit_corr = rdkit_df.drop(columns=['Delta_PCE']).corrwith(rdkit_df['Delta_PCE']).sort_values(key=abs, ascending=False)
        rdkit_corr.to_csv(f'{OUTPUT_DIR}/tables/rdkit_correlations.csv')
        print(f"  Calculated extended descriptors for {len(rdkit_data)} molecules")
        print(f"  Top correlations:")
        for feat, corr in rdkit_corr.head(5).items():
            print(f"    {feat}: {corr:.4f}")

    # Generate ECFP4 fingerprints
    print("\n[Step 6] Generating ECFP fingerprints...")
    ecfp_data = []
    for i, smiles in enumerate(df_clean['smiles'].values):
        if pd.notna(smiles):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                ecfp_data.append(arr)

    if ecfp_data:
        ecfp_array = np.array(ecfp_data)
        np.save(f'{OUTPUT_DIR}/fingerprints/ecfp4_fingerprints.npy', ecfp_array)
        print(f"  Saved ECFP4 fingerprints: {ecfp_array.shape}")

    # Generate MACCS fingerprints
    print("\n[Step 7] Generating MACCS fingerprints...")
    from rdkit.Chem import MACCSkeys
    maccs_data = []
    for smiles in df_clean['smiles'].values:
        if pd.notna(smiles):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol:
                fp = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                maccs_data.append(arr)

    if maccs_data:
        maccs_array = np.array(maccs_data)
        np.save(f'{OUTPUT_DIR}/fingerprints/maccs_fingerprints.npy', maccs_array)
        print(f"  Saved MACCS fingerprints: {maccs_array.shape}")

except ImportError:
    print("\n[Step 5-7] RDKit not available, skipping extended descriptors")

# ============================================================================
# STEP 5: Machine Learning Baseline
# ============================================================================
print("\n[Step 8] Building ML baseline models...")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Prepare data
X = df_clean[chemical_features].values
y = df_clean['Delta_PCE'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

# Train and evaluate models
results = []
for name, model in models.items():
    if name == 'SVR':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Cross-validation
    if name == 'SVR':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    results.append({
        'Model': name,
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'CV_R2_Mean': cv_scores.mean(),
        'CV_R2_Std': cv_scores.std()
    })
    print(f"  {name}: R2={results[-1]['R2']:.4f}, RMSE={results[-1]['RMSE']:.4f}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUTPUT_DIR}/tables/ml_model_comparison.csv', index=False)

# Feature importance from Random Forest
rf_model = models['Random Forest']
importance_df = pd.DataFrame({
    'Feature': chemical_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
importance_df.to_csv(f'{OUTPUT_DIR}/tables/feature_importance.csv', index=False)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Feature Importance from Random Forest', fontsize=14)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig5_feature_importance.png")

# ============================================================================
# STEP 6: Generate Summary Report
# ============================================================================
print("\n[Step 9] Generating summary report...")

report = f"""
================================================================================
QSPR ANALYSIS REPORT FOR PEROVSKITE SOLAR CELL MODULATORS
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: {INPUT_FILE}

================================================================================
1. DATA SUMMARY
================================================================================
- Initial samples: {initial_samples}
- After cleaning: {final_samples} ({100*final_samples/initial_samples:.2f}% retained)
- Chemical features: {', '.join(chemical_features)}
- Target: Delta_PCE = jv_reverse_scan_pce - jv_reverse_scan_pce_without_modulator

================================================================================
2. TARGET VARIABLE (Delta_PCE) STATISTICS
================================================================================
{pce_stats.to_string()}

- Skewness: {df_clean['Delta_PCE'].skew():.4f}
- Kurtosis: {df_clean['Delta_PCE'].kurtosis():.4f}
- Positive values: {(df_clean['Delta_PCE'] > 0).sum()} ({100*(df_clean['Delta_PCE'] > 0).sum()/final_samples:.1f}%)
- Negative values: {(df_clean['Delta_PCE'] < 0).sum()} ({100*(df_clean['Delta_PCE'] < 0).sum()/final_samples:.1f}%)

================================================================================
3. CORRELATIONS WITH DELTA_PCE
================================================================================
{corr_with_pce.to_string()}

================================================================================
4. ML MODEL COMPARISON
================================================================================
{results_df.to_string()}

================================================================================
5. TOP FEATURES BY IMPORTANCE (Random Forest)
================================================================================
{importance_df.to_string()}

================================================================================
6. OUTPUT FILES
================================================================================
- processed_data.csv: Cleaned dataset
- tables/: Statistical tables (CSV)
- figures/: Visualization plots (PNG)
- fingerprints/: Molecular fingerprints (NPY, CSV)
- critical_review.txt: Scientific review

================================================================================
END OF REPORT
================================================================================
"""

with open(f'{OUTPUT_DIR}/report/analysis_summary.txt', 'w') as f:
    f.write(report)

print(f"Saved report to {OUTPUT_DIR}/report/analysis_summary.txt")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
