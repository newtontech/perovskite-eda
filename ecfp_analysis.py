#!/usr/bin/env python3
"""
ECFP (Extended Connectivity Fingerprints) Molecular Fingerprint Analysis for QSPR
Analyzes the relationship between molecular fingerprints and Delta_PCE
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import umap
import os

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx'
OUTPUT_DIR = '/share/yhm/test/AutoML_EDA'
FINGERPRINT_DIR = os.path.join(OUTPUT_DIR, 'fingerprints')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Ensure directories exist
os.makedirs(FINGERPRINT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

print("=" * 80)
print("ECFP Molecular Fingerprint Analysis for QSPR")
print("=" * 80)

# ============================================================================
# Step 1: Load data and extract SMILES strings
# ============================================================================
print("\n[Step 1] Loading data and extracting SMILES strings...")

df = pd.read_excel(DATA_PATH)
print(f"  Total records: {len(df)}")
print(f"  Columns: {list(df.columns)}")

# Check SMILES column
if 'smiles' not in df.columns:
    raise ValueError("SMILES column not found in the dataset")

# Convert PCE columns to numeric
df['jv_reverse_scan_pce'] = pd.to_numeric(df['jv_reverse_scan_pce'], errors='coerce')
df['jv_reverse_scan_pce_without_modulator'] = pd.to_numeric(df['jv_reverse_scan_pce_without_modulator'], errors='coerce')

# Calculate Delta_PCE
df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']

# Filter valid entries
valid_mask = df['smiles'].notna() & df['Delta_PCE'].notna()
df_valid = df[valid_mask].copy()
print(f"  Valid records with SMILES and Delta_PCE: {len(df_valid)}")
print(f"  Delta_PCE statistics:")
print(f"    Mean: {df_valid['Delta_PCE'].mean():.4f}")
print(f"    Std:  {df_valid['Delta_PCE'].std():.4f}")
print(f"    Min:  {df_valid['Delta_PCE'].min():.4f}")
print(f"    Max:  {df_valid['Delta_PCE'].max():.4f}")

# ============================================================================
# Step 2: Generate ECFP fingerprints
# ============================================================================
print("\n[Step 2] Generating ECFP fingerprints using RDKit...")

def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
    """Convert SMILES to ECFP fingerprint"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Generate ECFP4 (radius=2)
print("  Generating ECFP4 (radius=2, 2048 bits)...")
ecfp4_list = []
valid_indices = []

for idx, row in df_valid.iterrows():
    fp = smiles_to_ecfp(row['smiles'], radius=2, n_bits=2048)
    if fp is not None:
        ecfp4_list.append(fp)
        valid_indices.append(idx)

ecfp4 = np.array(ecfp4_list)
print(f"  ECFP4 shape: {ecfp4.shape}")

# Generate ECFP6 (radius=3)
print("  Generating ECFP6 (radius=3, 2048 bits)...")
ecfp6_list = []

for idx in valid_indices:
    smiles = df_valid.loc[idx, 'smiles']
    fp = smiles_to_ecfp(smiles, radius=3, n_bits=2048)
    if fp is not None:
        ecfp6_list.append(fp)

ecfp6 = np.array(ecfp6_list)
print(f"  ECFP6 shape: {ecfp6.shape}")

# Get corresponding Delta_PCE values
delta_pce = df_valid.loc[valid_indices, 'Delta_PCE'].values

# Save fingerprints
ecfp4_df = pd.DataFrame(ecfp4, columns=[f'Bit_{i}' for i in range(2048)])
ecfp4_df['Delta_PCE'] = delta_pce
ecfp4_df['SMILES'] = df_valid.loc[valid_indices, 'smiles'].values
ecfp4_df.to_csv(os.path.join(FINGERPRINT_DIR, 'ecfp_fingerprints.csv'), index=False)
print(f"  Fingerprints saved to: {FINGERPRINT_DIR}/ecfp_fingerprints.csv")

# ============================================================================
# Step 3: Analyze fingerprint bits vs Delta_PCE
# ============================================================================
print("\n[Step 3] Analyzing fingerprint bits vs Delta_PCE...")

# Calculate bit frequencies
bit_frequencies = ecfp4.sum(axis=0) / ecfp4.shape[0]
print(f"  Bit frequency range: {bit_frequencies.min():.4f} - {bit_frequencies.max():.4f}")
print(f"  Active bits (>0): {(bit_frequencies > 0).sum()}")

# Point-biserial correlation for each bit
print("  Computing point-biserial correlations...")
correlations = []
p_values_corr = []

for i in range(2048):
    if bit_frequencies[i] > 0 and bit_frequencies[i] < 1:  # Skip constant bits
        corr, p_val = pointbiserialr(ecfp4[:, i], delta_pce)
        correlations.append(corr)
        p_values_corr.append(p_val)
    else:
        correlations.append(np.nan)
        p_values_corr.append(np.nan)

correlations = np.array(correlations)
p_values_corr = np.array(p_values_corr)

# Chi-square test (binarize Delta_PCE at median)
print("  Computing chi-square tests...")
median_delta = np.median(delta_pce)
delta_binary = (delta_pce > median_delta).astype(int)

chi2_stats = []
chi2_pvalues = []

for i in range(2048):
    if bit_frequencies[i] > 0 and bit_frequencies[i] < 1:
        contingency = pd.crosstab(ecfp4[:, i], delta_binary)
        if contingency.shape == (2, 2):
            chi2, p_val, dof, expected = chi2_contingency(contingency)
            chi2_stats.append(chi2)
            chi2_pvalues.append(p_val)
        else:
            chi2_stats.append(np.nan)
            chi2_pvalues.append(np.nan)
    else:
        chi2_stats.append(np.nan)
        chi2_pvalues.append(np.nan)

chi2_stats = np.array(chi2_stats)
chi2_pvalues = np.array(chi2_pvalues)

# Compile bit importance results
bit_importance = pd.DataFrame({
    'Bit': range(2048),
    'Frequency': bit_frequencies,
    'Correlation': correlations,
    'Correlation_PValue': p_values_corr,
    'Chi2_Stat': chi2_stats,
    'Chi2_PValue': chi2_pvalues
})

# Add significance flags
bit_importance['Significant_Corr'] = (bit_importance['Correlation_PValue'] < 0.05) & (~bit_importance['Correlation_PValue'].isna())
bit_importance['Significant_Chi2'] = (bit_importance['Chi2_PValue'] < 0.05) & (~bit_importance['Chi2_PValue'].isna())

# Sort by absolute correlation
bit_importance['Abs_Correlation'] = bit_importance['Correlation'].abs()
bit_importance_sorted = bit_importance.sort_values('Abs_Correlation', ascending=False)

# Save bit importance
bit_importance_sorted.to_csv(os.path.join(FINGERPRINT_DIR, 'ecfp_bit_importance.csv'), index=False)
print(f"  Bit importance saved to: {FINGERPRINT_DIR}/ecfp_bit_importance.csv")

# Print top 20 significant bits
print("\n  Top 20 most significant bits (by correlation):")
print("-" * 80)
top20 = bit_importance_sorted.head(20)
for idx, row in top20.iterrows():
    print(f"  Bit {int(row['Bit']):4d}: Freq={row['Frequency']:.4f}, "
          f"Corr={row['Correlation']:+.4f}, P={row['Correlation_PValue']:.2e}, "
          f"Chi2={row['Chi2_Stat']:.2f}, Chi2-P={row['Chi2_PValue']:.2e}")

print(f"\n  Significant bits (correlation p<0.05): {bit_importance['Significant_Corr'].sum()}")
print(f"  Significant bits (chi-square p<0.05): {bit_importance['Significant_Chi2'].sum()}")

# ============================================================================
# Step 4: Dimensionality reduction
# ============================================================================
print("\n[Step 4] Performing dimensionality reduction...")

# t-SNE on ECFP4
print("  Computing t-SNE (this may take a while)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(ecfp4)//4))
tsne_result = tsne.fit_transform(ecfp4)

# UMAP
print("  Computing UMAP...")
reducer = umap.UMAP(random_state=42, n_neighbors=min(15, len(ecfp4)//4), min_spread=0.5)
umap_result = reducer.fit_transform(ecfp4)

# Plot t-SNE
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# t-SNE colored by Delta_PCE
sc1 = axes[0].scatter(tsne_result[:, 0], tsne_result[:, 1],
                      c=delta_pce, cmap='RdYlBu_r', alpha=0.7, s=30)
axes[0].set_xlabel('t-SNE 1', fontsize=12)
axes[0].set_ylabel('t-SNE 2', fontsize=12)
axes[0].set_title('t-SNE of ECFP4 Fingerprints\nColored by Delta_PCE', fontsize=14)
plt.colorbar(sc1, ax=axes[0], label='Delta_PCE')

# UMAP colored by Delta_PCE
sc2 = axes[1].scatter(umap_result[:, 0], umap_result[:, 1],
                      c=delta_pce, cmap='RdYlBu_r', alpha=0.7, s=30)
axes[1].set_xlabel('UMAP 1', fontsize=12)
axes[1].set_ylabel('UMAP 2', fontsize=12)
axes[1].set_title('UMAP of ECFP4 Fingerprints\nColored by Delta_PCE', fontsize=14)
plt.colorbar(sc2, ax=axes[1], label='Delta_PCE')

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'ecfp_tsne.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  t-SNE and UMAP plot saved to: {FIGURE_DIR}/ecfp_tsne.png")

# Individual UMAP plot
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(umap_result[:, 0], umap_result[:, 1],
                c=delta_pce, cmap='RdYlBu_r', alpha=0.7, s=40)
ax.set_xlabel('UMAP 1', fontsize=14)
ax.set_ylabel('UMAP 2', fontsize=14)
ax.set_title('UMAP Visualization of ECFP4 Fingerprints\nColored by Delta_PCE', fontsize=16)
plt.colorbar(sc, ax=ax, label='Delta_PCE')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'ecfp_umap.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  UMAP plot saved to: {FIGURE_DIR}/ecfp_umap.png")

# ============================================================================
# Step 5: Build ML models
# ============================================================================
print("\n[Step 5] Building and evaluating ML models...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    ecfp4, delta_pce, test_size=0.2, random_state=42
)
print(f"  Training set: {len(X_train)} samples")
print(f"  Test set: {len(X_test)} samples")

results = []

# Random Forest
print("\n  [5.1] Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred_train = rf_model.predict(X_train)
rf_pred_test = rf_model.predict(X_test)

rf_results = {
    'Model': 'Random Forest',
    'Train_R2': r2_score(y_train, rf_pred_train),
    'Test_R2': r2_score(y_test, rf_pred_test),
    'Test_RMSE': np.sqrt(mean_squared_error(y_test, rf_pred_test)),
    'Test_MAE': mean_absolute_error(y_test, rf_pred_test),
    'CV_R2_Mean': cross_val_score(rf_model, ecfp4, delta_pce, cv=5, scoring='r2').mean(),
    'CV_R2_Std': cross_val_score(rf_model, ecfp4, delta_pce, cv=5, scoring='r2').std()
}
results.append(rf_results)

print(f"    Train R2: {rf_results['Train_R2']:.4f}")
print(f"    Test R2:  {rf_results['Test_R2']:.4f}")
print(f"    Test RMSE: {rf_results['Test_RMSE']:.4f}")
print(f"    Test MAE:  {rf_results['Test_MAE']:.4f}")
print(f"    CV R2:     {rf_results['CV_R2_Mean']:.4f} (+/- {rf_results['CV_R2_Std']:.4f})")

# Gradient Boosting
print("\n  [5.2] Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
gb_model.fit(X_train, y_train)
gb_pred_train = gb_model.predict(X_train)
gb_pred_test = gb_model.predict(X_test)

gb_results = {
    'Model': 'Gradient Boosting',
    'Train_R2': r2_score(y_train, gb_pred_train),
    'Test_R2': r2_score(y_test, gb_pred_test),
    'Test_RMSE': np.sqrt(mean_squared_error(y_test, gb_pred_test)),
    'Test_MAE': mean_absolute_error(y_test, gb_pred_test),
    'CV_R2_Mean': cross_val_score(gb_model, ecfp4, delta_pce, cv=5, scoring='r2').mean(),
    'CV_R2_Std': cross_val_score(gb_model, ecfp4, delta_pce, cv=5, scoring='r2').std()
}
results.append(gb_results)

print(f"    Train R2: {gb_results['Train_R2']:.4f}")
print(f"    Test R2:  {gb_results['Test_R2']:.4f}")
print(f"    Test RMSE: {gb_results['Test_RMSE']:.4f}")
print(f"    Test MAE:  {gb_results['Test_MAE']:.4f}")
print(f"    CV R2:     {gb_results['CV_R2_Mean']:.4f} (+/- {gb_results['CV_R2_Std']:.4f})")

# Save ML results
ml_results_df = pd.DataFrame(results)
ml_results_df.to_csv(os.path.join(FINGERPRINT_DIR, 'ecfp_ml_results.csv'), index=False)
print(f"\n  ML results saved to: {FINGERPRINT_DIR}/ecfp_ml_results.csv")

# Feature importance from Random Forest
rf_importance = pd.DataFrame({
    'Bit': range(2048),
    'RF_Importance': rf_model.feature_importances_
}).sort_values('RF_Importance', ascending=False)

print("\n  Top 20 most important bits (Random Forest):")
print("-" * 40)
for idx, row in rf_importance.head(20).iterrows():
    print(f"    Bit {int(row['Bit']):4d}: {row['RF_Importance']:.6f}")

# ============================================================================
# Step 6: Generate summary report
# ============================================================================
print("\n[Step 6] Generating summary report...")

summary_lines = []
summary_lines.append("=" * 80)
summary_lines.append("ECFP MOLECULAR FINGERPRINT ANALYSIS FOR QSPR")
summary_lines.append("=" * 80)
summary_lines.append(f"\nDate: 2026-02-20")
summary_lines.append(f"Data Source: {DATA_PATH}")
summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("1. DATA SUMMARY")
summary_lines.append("-" * 80)
summary_lines.append(f"Total records in dataset: {len(df)}")
summary_lines.append(f"Valid records with SMILES and Delta_PCE: {len(df_valid)}")
summary_lines.append(f"Records successfully converted to ECFP: {len(valid_indices)}")
summary_lines.append("")
summary_lines.append("Delta_PCE Statistics:")
summary_lines.append(f"  Mean:   {df_valid['Delta_PCE'].mean():.4f}")
summary_lines.append(f"  Std:    {df_valid['Delta_PCE'].std():.4f}")
summary_lines.append(f"  Min:    {df_valid['Delta_PCE'].min():.4f}")
summary_lines.append(f"  Max:    {df_valid['Delta_PCE'].max():.4f}")
summary_lines.append(f"  Median: {df_valid['Delta_PCE'].median():.4f}")
summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("2. FINGERPRINT GENERATION")
summary_lines.append("-" * 80)
summary_lines.append(f"ECFP4 (Morgan, radius=2): {ecfp4.shape[1]} bits, {ecfp4.shape[0]} samples")
summary_lines.append(f"ECFP6 (Morgan, radius=3): {ecfp6.shape[1]} bits, {ecfp6.shape[0]} samples")
summary_lines.append(f"Active bits in ECFP4: {(bit_frequencies > 0).sum()} / 2048")
summary_lines.append(f"Bit frequency range: {bit_frequencies.min():.4f} - {bit_frequencies.max():.4f}")
summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("3. STATISTICAL ANALYSIS")
summary_lines.append("-" * 80)
summary_lines.append(f"Significant bits by correlation (p<0.05): {bit_importance['Significant_Corr'].sum()}")
summary_lines.append(f"Significant bits by chi-square (p<0.05): {bit_importance['Significant_Chi2'].sum()}")
summary_lines.append("")
summary_lines.append("Top 10 bits by absolute correlation:")
summary_lines.append("-" * 60)
for i, (idx, row) in enumerate(bit_importance_sorted.head(10).iterrows()):
    summary_lines.append(
        f"  {i+1:2d}. Bit {int(row['Bit']):4d}: r={row['Correlation']:+.4f}, "
        f"p={row['Correlation_PValue']:.2e}, freq={row['Frequency']:.4f}"
    )
summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("4. MACHINE LEARNING RESULTS")
summary_lines.append("-" * 80)
summary_lines.append(f"{'Model':<20} {'Train R2':>10} {'Test R2':>10} {'RMSE':>10} {'MAE':>10} {'CV R2':>12}")
summary_lines.append("-" * 80)
for r in results:
    summary_lines.append(
        f"{r['Model']:<20} {r['Train_R2']:>10.4f} {r['Test_R2']:>10.4f} "
        f"{r['Test_RMSE']:>10.4f} {r['Test_MAE']:>10.4f} "
        f"{r['CV_R2_Mean']:>8.4f}+/-{r['CV_R2_Std']:.3f}"
    )
summary_lines.append("")

# Best model
best_model = max(results, key=lambda x: x['CV_R2_Mean'])
summary_lines.append(f"Best model by CV R2: {best_model['Model']} (CV R2 = {best_model['CV_R2_Mean']:.4f})")
summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("5. TOP FEATURES (Random Forest Importance)")
summary_lines.append("-" * 80)
for i, (idx, row) in enumerate(rf_importance.head(20).iterrows()):
    summary_lines.append(f"  {i+1:2d}. Bit {int(row['Bit']):4d}: {row['RF_Importance']:.6f}")
summary_lines.append("")

summary_lines.append("-" * 80)
summary_lines.append("6. OUTPUT FILES")
summary_lines.append("-" * 80)
summary_lines.append(f"ECFP Fingerprints:  {FINGERPRINT_DIR}/ecfp_fingerprints.csv")
summary_lines.append(f"Bit Importance:     {FINGERPRINT_DIR}/ecfp_bit_importance.csv")
summary_lines.append(f"ML Results:         {FINGERPRINT_DIR}/ecfp_ml_results.csv")
summary_lines.append(f"t-SNE Plot:         {FIGURE_DIR}/ecfp_tsne.png")
summary_lines.append(f"UMAP Plot:          {FIGURE_DIR}/ecfp_umap.png")
summary_lines.append(f"This Summary:       {FINGERPRINT_DIR}/ecfp_analysis.txt")
summary_lines.append("")

summary_lines.append("=" * 80)
summary_lines.append("ANALYSIS COMPLETE")
summary_lines.append("=" * 80)

summary_text = "\n".join(summary_lines)

with open(os.path.join(FINGERPRINT_DIR, 'ecfp_analysis.txt'), 'w') as f:
    f.write(summary_text)

print(summary_text)
print("\n" + "=" * 80)
print("ECFP Analysis Complete!")
print("=" * 80)
