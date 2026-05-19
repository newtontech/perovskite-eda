#!/usr/bin/env python3
"""
MACCS Fingerprint Analysis for QSPR
Analyzes Molecular ACCess System (MACCS) fingerprints against Delta_PCE
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, Draw
from rdkit.Chem.Draw import MolsToGridImage
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pointbiserialr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("MACCS FINGERPRINT ANALYSIS FOR QSPR")
print("="*80)

# ============================================================================
# 1. Load Data and Extract SMILES
# ============================================================================
print("\n1. Loading data...")
data_path = '/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx'
df = pd.read_excel(data_path)
print(f"   Total rows loaded: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# Extract relevant columns
smiles_col = 'smiles'
pce_col = 'jv_reverse_scan_pce'
pce_no_mod_col = 'jv_reverse_scan_pce_without_modulator'

# Convert PCE columns to numeric
df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
df[pce_no_mod_col] = pd.to_numeric(df[pce_no_mod_col], errors='coerce')

# Filter valid data
df_clean = df[[smiles_col, pce_col, pce_no_mod_col]].dropna()
print(f"   Rows with valid SMILES and PCE data: {len(df_clean)}")

# ============================================================================
# 2. Calculate Delta_PCE
# ============================================================================
print("\n2. Calculating Delta_PCE...")
df_clean['Delta_PCE'] = df_clean[pce_col] - df_clean[pce_no_mod_col]
print(f"   Delta_PCE statistics:")
print(f"   - Mean: {df_clean['Delta_PCE'].mean():.4f}")
print(f"   - Std: {df_clean['Delta_PCE'].std():.4f}")
print(f"   - Min: {df_clean['Delta_PCE'].min():.4f}")
print(f"   - Max: {df_clean['Delta_PCE'].max():.4f}")
print(f"   - Positive Delta_PCE count: {(df_clean['Delta_PCE'] > 0).sum()}")
print(f"   - Negative Delta_PCE count: {(df_clean['Delta_PCE'] < 0).sum()}")

# ============================================================================
# 3. Generate MACCS Fingerprints
# ============================================================================
print("\n3. Generating MACCS fingerprints (166 bits)...")

def generate_maccs_fingerprint(smiles):
    """Generate MACCS fingerprint from SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    # Convert to bit vector (keys 1-166, index 0 is unused)
    return [int(fp.GetBit(i)) for i in range(1, 167)]

# Generate fingerprints
fingerprints = []
valid_indices = []
for idx, row in df_clean.iterrows():
    fp = generate_maccs_fingerprint(row[smiles_col])
    if fp is not None:
        fingerprints.append(fp)
        valid_indices.append(idx)

# Create fingerprint dataframe
fp_df = pd.DataFrame(fingerprints, columns=[f'MACCS_{i}' for i in range(1, 167)])
fp_df.index = valid_indices

# Merge with Delta_PCE
analysis_df = df_clean.loc[valid_indices].copy()
analysis_df = pd.concat([analysis_df, fp_df], axis=1)

print(f"   Successfully generated fingerprints: {len(fingerprints)}")

# Save fingerprints
fp_output_path = '/share/yhm/test/AutoML_EDA/fingerprints/maccs_fingerprints.csv'
fp_df.to_csv(fp_output_path)
print(f"   Fingerprints saved to: {fp_output_path}")

# ============================================================================
# 4. Analyze MACCS Keys vs Delta_PCE
# ============================================================================
print("\n4. Analyzing MACCS keys vs Delta_PCE...")

maccs_cols = [f'MACCS_{i}' for i in range(1, 167)]
correlations = []
p_values = []

for col in maccs_cols:
    # Point-biserial correlation
    corr, p_val = pointbiserialr(analysis_df[col], analysis_df['Delta_PCE'])
    correlations.append(corr)
    p_values.append(p_val)

# Create correlation dataframe
corr_df = pd.DataFrame({
    'MACCS_Key': range(1, 167),
    'Correlation': correlations,
    'P_Value': p_values,
    'Abs_Correlation': [abs(c) for c in correlations],
    'Significant': [p < 0.05 for p in p_values]
})

# Sort by absolute correlation
corr_df_sorted = corr_df.sort_values('Abs_Correlation', ascending=False)

# Save correlation results
corr_output_path = '/share/yhm/test/AutoML_EDA/fingerprints/maccs_key_correlation.csv'
corr_df_sorted.to_csv(corr_output_path, index=False)
print(f"   Correlation results saved to: {corr_output_path}")

# Top 20 most correlated keys
top_20 = corr_df_sorted.head(20)
print("\n   Top 20 MACCS keys by absolute correlation:")
print("   " + "-"*60)
for _, row in top_20.iterrows():
    sig = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
    print(f"   Key {int(row['MACCS_Key']):3d}: r = {row['Correlation']:+.4f}, p = {row['P_Value']:.4e} {sig}")

# ============================================================================
# 5. Visualizations
# ============================================================================
print("\n5. Creating visualizations...")

# MACCS Key definitions (simplified - common substructures)
MACCS_DEFINITIONS = {
    1: "Isotope",
    2: "Atomic number > 85",
    3: "Atomic number > 50",
    4: "Atomic number > 35",
    5: "Atomic number > 20",
    6: "Atomic number > 17",
    7: "Atomic number > 14",
    8: "Atomic number > 10",
    9: "Elements: Li, Na, K, Ca, Mg",
    10: "Elements: B, Si, P",
    11: "Elements: F, Cl, Br, I",
    12: "Number of heavy atoms >= 22",
    13: "Number of heavy atoms >= 18",
    14: "Number of heavy atoms >= 16",
    15: "Number of heavy atoms >= 14",
    16: "Number of heavy atoms >= 12",
    17: "Number of heavy atoms >= 10",
    18: "Number of heavy atoms >= 8",
    19: "Number of heavy atoms >= 6",
    20: "Number of heavy atoms >= 4",
    21: "Aromatic ring count >= 2",
    22: "Aromatic ring count >= 1",
    23: "Heteroaromatic ring",
    24: "Heterocyclic ring",
    25: "Number of rings >= 3",
    26: "Number of rings >= 2",
    27: "Number of rings >= 1",
    28: "Number of rings >= 4",
    29: "Number of rings >= 5",
    30: "Number of rings >= 6",
    31: "Aromatics >= 50%",
    32: "Aromatics >= 40%",
    33: "Aromatics >= 30%",
    34: "Aromatics >= 20%",
    35: "Aromatics >= 10%",
    36: "Aromatics >= 0%",
    37: "Aromatics >= 0% (different calculation)",
    38: "4-membered ring",
    39: "5-membered ring",
    40: "6-membered ring",
    41: "7-membered ring",
    42: "8-membered ring",
    43: "Carbocyclic ring",
    44: "Multiple aromatic rings",
    45: "Fused aromatic rings",
    46: "Fused rings",
    47: "Fused aromatic rings (2)",
    48: "Bicyclic",
    49: "Bridgehead atoms",
    50: "Spiro atoms",
    51: "Saturated or aromatic carbocycles",
    52: "Saturated or aromatic heterocycles",
    53: "Saturated carbocycle",
    54: "Saturated heterocycle",
    55: "Unsaturated carbocycle",
    56: "Unsaturated heterocycle",
    57: "Unsaturated heterocycle (2)",
    58: "Aromatic heterocycle",
    59: "2+ fused aromatic rings",
    60: "2+ fused saturated rings",
    61: "2+ fused rings",
    62: "CH3",
    63: "CH2",
    64: "CH",
    65: "Quaternary C",
    66: "C=C",
    67: "C#C",
    68: "C=C-C=C",
    69: "C=C-C=C-C=C",
    70: "C1=CC=CC=C1 (Benzene)",
    71: "Aromatic C",
    72: "Aromatic N",
    73: "Aromatic O",
    74: "Aromatic S",
    75: "Aromatic hetero atom",
    76: ">=2 aromatic rings",
    77: ">=3 aromatic rings",
    78: ">=4 aromatic rings",
    79: ">=5 aromatic rings",
    80: "Isotope (C)",
    81: "Isotope (O)",
    82: "Isotope (N)",
    83: "Isotope (S)",
    84: "Any atom with isotope",
    85: "H attached to C(sp3)",
    86: "H attached to C(sp2)",
    87: "H attached to hetero",
    88: "Any H attached to hetero",
    89: "H attached to Group 15/16",
    90: "H attached to Group 15/16 (sp3)",
    91: "H attached to hetero (sp3)",
    92: "H attached to hetero (sp2)",
    93: "H attached to hetero (aromatic)",
    94: ">=3 H on hetero",
    95: ">=2 H on hetero",
    96: ">=1 H on hetero",
    97: "OH",
    98: "SH",
    99: "NH2",
    100: "NH",
    101: "Aliphatic alcohol -OH",
    102: "Phenol -OH",
    103: "Carboxylic acid -OH",
    104: "Enol -OH",
    105: "Carboxylic acid derivative",
    106: "Ester -O-",
    107: "Ether -O-",
    108: "Aldehyde -CHO",
    109: "Ketone >C=O",
    110: "Carboxylic acid -COOH",
    111: "Carboxylic acid derivative (2)",
    112: "Amide -CON<",
    113: "Primary amine -NH2",
    114: "Secondary amine -NH-",
    115: "Tertiary amine -N<",
    116: "N in aromatic ring",
    117: "N in non-aromatic ring",
    118: "N with 4 bonds",
    119: "N with double bond",
    120: "Imine >C=N-",
    121: "Nitrile -C#N",
    122: "N in nitro group",
    123: "Azide -N3",
    124: "Sulfide -S-",
    125: "Thiol -SH",
    126: "Disulfide -S-S-",
    127: "Sulfoxide S=O",
    128: "Sulfone O=S=O",
    129: "Sulfonic acid -SO3H",
    130: "Sulfonate -SO3-",
    131: "Phosphorus present",
    132: "Phosphine/phosphine oxide",
    133: "Phosphate",
    134: "Halogen",
    135: "Fluorine",
    136: "Chlorine",
    137: "Bromine",
    138: "Iodine",
    139: "Cyclic ether",
    140: "Epoxide",
    141: "N in 3-membered ring",
    142: "N in 4-membered ring",
    143: "N in 5-membered ring",
    144: "N in 6-membered ring",
    145: "O in 5-membered ring",
    146: "O in 6-membered ring",
    147: "S in ring",
    148: "1,2-diol",
    149: "1,3-diol",
    150: "Enol ether",
    151: "Vinyl (CH2=CH-)",
    152: "Carbamate -OC(=O)N",
    153: "Urea -NC(=O)N",
    154: "Carbamate-like",
    155: "Guanidine",
    156: "Guanidinium",
    157: "Imidazole",
    158: "Pyridine",
    159: "Furan",
    160: "Thiophene",
    161: "Pyrrole",
    162: "Pyrimidine",
    163: "Quaternary N+",
    164: "Positive charge",
    165: "Negative charge",
    166: "Miscellaneous"
}

# 5.1 Bar plot of top MACCS keys correlation
fig, ax = plt.subplots(figsize=(14, 8))
top_20_plot = top_20.copy()
top_20_plot['Label'] = top_20_plot['MACCS_Key'].apply(
    lambda x: f"Key {int(x)}: {MACCS_DEFINITIONS.get(int(x), 'Unknown')[:30]}"
)

colors = ['#e74c3c' if c > 0 else '#3498db' for c in top_20_plot['Correlation']]
bars = ax.barh(range(len(top_20_plot)), top_20_plot['Correlation'], color=colors)
ax.set_yticks(range(len(top_20_plot)))
ax.set_yticklabels(top_20_plot['Label'])
ax.set_xlabel('Point-Biserial Correlation with Delta_PCE', fontsize=12)
ax.set_title('Top 20 MACCS Keys Correlated with Delta_PCE', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Add significance markers
for i, (_, row) in enumerate(top_20_plot.iterrows()):
    if row['P_Value'] < 0.001:
        ax.text(row['Correlation'], i, ' ***', ha='left' if row['Correlation'] > 0 else 'right', va='center', fontsize=8)
    elif row['P_Value'] < 0.01:
        ax.text(row['Correlation'], i, ' **', ha='left' if row['Correlation'] > 0 else 'right', va='center', fontsize=8)
    elif row['P_Value'] < 0.05:
        ax.text(row['Correlation'], i, ' *', ha='left' if row['Correlation'] > 0 else 'right', va='center', fontsize=8)

plt.tight_layout()
top_keys_path = '/share/yhm/test/AutoML_EDA/figures/maccs_top_keys.png'
plt.savefig(top_keys_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   Top keys plot saved to: {top_keys_path}")

# 5.2 Heatmap of MACCS keys presence by Delta_PCE quartile
fig, ax = plt.subplots(figsize=(16, 10))

# Create quartiles
analysis_df['PCE_Quartile'] = pd.qcut(analysis_df['Delta_PCE'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

# Select top 25 MACCS keys for heatmap
top_keys = top_20['MACCS_Key'].head(25).astype(int).tolist()
top_maccs_cols = [f'MACCS_{k}' for k in top_keys]

# Calculate mean presence for each quartile
quartile_means = analysis_df.groupby('PCE_Quartile')[top_maccs_cols].mean()

# Rename columns for display
quartile_means.columns = [f"Key {k}: {MACCS_DEFINITIONS.get(k, 'Unknown')[:20]}" for k in top_keys]

sns.heatmap(quartile_means.T, annot=True, fmt='.2f', cmap='RdYlBu_r',
            cbar_kws={'label': 'Proportion Present'}, ax=ax)
ax.set_title('MACCS Key Presence by Delta_PCE Quartile', fontsize=14, fontweight='bold')
ax.set_xlabel('Delta_PCE Quartile')
ax.set_ylabel('MACCS Key')
plt.tight_layout()
heatmap_path = '/share/yhm/test/AutoML_EDA/figures/maccs_heatmap.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   Heatmap saved to: {heatmap_path}")

# 5.3 PCA on MACCS fingerprints colored by Delta_PCE
print("   Performing PCA...")
X = fp_df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=analysis_df['Delta_PCE'].values,
                     cmap='coolwarm', alpha=0.6, s=50, edgecolors='black', linewidth=0.3)
cbar = plt.colorbar(scatter)
cbar.set_label('Delta_PCE', fontsize=12)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax.set_title('PCA of MACCS Fingerprints colored by Delta_PCE', fontsize=14, fontweight='bold')
plt.tight_layout()
pca_path = '/share/yhm/test/AutoML_EDA/figures/maccs_pca.png'
plt.savefig(pca_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   PCA plot saved to: {pca_path}")

# ============================================================================
# 6. Build ML Models
# ============================================================================
print("\n6. Building ML models...")

# Create binary target: positive vs negative Delta_PCE
analysis_df['Delta_PCE_Class'] = (analysis_df['Delta_PCE'] > 0).astype(int)
y = analysis_df['Delta_PCE_Class'].values
X = fp_df.values

print(f"   Positive class (Delta_PCE > 0): {sum(y)} samples")
print(f"   Negative class (Delta_PCE <= 0): {len(y) - sum(y)} samples")

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6.1 Random Forest Classifier
print("\n   6.1 Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='roc_auc')
rf_model.fit(X, y)
rf_feature_importance = rf_model.feature_importances_

print(f"       ROC-AUC (CV): {rf_scores.mean():.4f} (+/- {rf_scores.std()*2:.4f})")

# 6.2 Logistic Regression with L1 regularization
print("\n   6.2 Logistic Regression (L1)...")
lr_model = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=1000, random_state=42)
lr_scores = cross_val_score(lr_model, X, y, cv=cv, scoring='roc_auc')
lr_model.fit(X_scaled, y)
lr_coefficients = lr_model.coef_[0]

print(f"       ROC-AUC (CV): {lr_scores.mean():.4f} (+/- {lr_scores.std()*2:.4f})")

# 6.3 ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

# RF ROC curve
rf_proba = rf_model.predict_proba(X)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y, rf_proba)
rf_auc = roc_auc_score(y, rf_proba)
ax.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})', linewidth=2)

# LR ROC curve
lr_proba = lr_model.predict_proba(X_scaled)[:, 1]
lr_fpr, lr_tpr, _ = roc_curve(y, lr_proba)
lr_auc = roc_auc_score(y, lr_proba)
ax.plot(lr_fpr, lr_tpr, label=f'Logistic Regression L1 (AUC = {lr_auc:.3f})', linewidth=2)

# Diagonal
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves for Delta_PCE Classification', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
roc_path = '/share/yhm/test/AutoML_EDA/figures/maccs_roc_curves.png'
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ROC curves saved to: {roc_path}")

# 6.4 Feature importance plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# RF feature importance
rf_imp_df = pd.DataFrame({
    'MACCS_Key': range(1, 167),
    'Importance': rf_feature_importance
}).sort_values('Importance', ascending=True).tail(15)
rf_imp_df['Label'] = rf_imp_df['MACCS_Key'].apply(
    lambda x: f"Key {int(x)}: {MACCS_DEFINITIONS.get(int(x), 'Unknown')[:25]}")
axes[0].barh(range(len(rf_imp_df)), rf_imp_df['Importance'], color='#3498db')
axes[0].set_yticks(range(len(rf_imp_df)))
axes[0].set_yticklabels(rf_imp_df['Label'])
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('Random Forest - Top 15 Features', fontweight='bold')

# LR coefficients
lr_coef_df = pd.DataFrame({
    'MACCS_Key': range(1, 167),
    'Coefficient': lr_coefficients
})
lr_coef_df['Abs_Coef'] = abs(lr_coef_df['Coefficient'])
lr_coef_df = lr_coef_df.sort_values('Abs_Coef', ascending=True).tail(15)
lr_coef_df['Label'] = lr_coef_df['MACCS_Key'].apply(
    lambda x: f"Key {int(x)}: {MACCS_DEFINITIONS.get(int(x), 'Unknown')[:25]}")
colors = ['#e74c3c' if c > 0 else '#3498db' for c in lr_coef_df['Coefficient']]
axes[1].barh(range(len(lr_coef_df)), lr_coef_df['Coefficient'], color=colors)
axes[1].set_yticks(range(len(lr_coef_df)))
axes[1].set_yticklabels(lr_coef_df['Label'])
axes[1].set_xlabel('Coefficient')
axes[1].set_title('Logistic Regression L1 - Top 15 Features', fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
importance_path = '/share/yhm/test/AutoML_EDA/figures/maccs_feature_importance.png'
plt.savefig(importance_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   Feature importance plot saved to: {importance_path}")

# Save ML results
ml_results = pd.DataFrame({
    'MACCS_Key': range(1, 167),
    'RF_Importance': rf_feature_importance,
    'LR_Coefficient': lr_coefficients,
    'PointBiserial_Corr': correlations,
    'PointBiserial_PVal': p_values
})
ml_results['RF_Rank'] = ml_results['RF_Importance'].rank(ascending=False)
ml_results['LR_Rank'] = ml_results['LR_Coefficient'].abs().rank(ascending=False)
ml_results = ml_results.sort_values('RF_Importance', ascending=False)
ml_output_path = '/share/yhm/test/AutoML_EDA/fingerprints/maccs_ml_results.csv'
ml_results.to_csv(ml_output_path, index=False)
print(f"   ML results saved to: {ml_output_path}")

# ============================================================================
# 7. Interpret Key MACCS Features
# ============================================================================
print("\n7. Generating interpretation report...")

interpretation_lines = []
interpretation_lines.append("="*80)
interpretation_lines.append("MACCS FINGERPRINT ANALYSIS - INTERPRETATION REPORT")
interpretation_lines.append("="*80)
interpretation_lines.append("")
interpretation_lines.append("OVERVIEW")
interpretation_lines.append("-"*40)
interpretation_lines.append(f"Total molecules analyzed: {len(analysis_df)}")
interpretation_lines.append(f"Delta_PCE range: {analysis_df['Delta_PCE'].min():.4f} to {analysis_df['Delta_PCE'].max():.4f}")
interpretation_lines.append(f"Positive Delta_PCE (improvement): {sum(y)} ({100*sum(y)/len(y):.1f}%)")
interpretation_lines.append(f"Negative Delta_PCE (degradation): {len(y)-sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")
interpretation_lines.append("")
interpretation_lines.append("MACHINE LEARNING PERFORMANCE")
interpretation_lines.append("-"*40)
interpretation_lines.append(f"Random Forest ROC-AUC: {rf_scores.mean():.4f} (+/- {rf_scores.std()*2:.4f})")
interpretation_lines.append(f"Logistic Regression L1 ROC-AUC: {lr_scores.mean():.4f} (+/- {lr_scores.std()*2:.4f})")
interpretation_lines.append("")

interpretation_lines.append("TOP 20 MACCS KEYS - STATISTICAL CORRELATION")
interpretation_lines.append("-"*60)
for rank, (_, row) in enumerate(top_20.iterrows(), 1):
    key = int(row['MACCS_Key'])
    desc = MACCS_DEFINITIONS.get(key, "Unknown substructure")
    interpretation_lines.append(f"")
    interpretation_lines.append(f"Rank {rank}: MACCS Key {key}")
    interpretation_lines.append(f"  Description: {desc}")
    interpretation_lines.append(f"  Correlation: {row['Correlation']:+.4f}")
    interpretation_lines.append(f"  P-value: {row['P_Value']:.4e}")
    interpretation_lines.append(f"  Significant: {'Yes' if row['Significant'] else 'No'}")

    if row['Correlation'] > 0:
        interpretation_lines.append(f"  Interpretation: Presence of this substructure tends to INCREASE Delta_PCE")
    else:
        interpretation_lines.append(f"  Interpretation: Presence of this substructure tends to DECREASE Delta_PCE")

interpretation_lines.append("")
interpretation_lines.append("TOP 10 FEATURES - RANDOM FOREST MODEL")
interpretation_lines.append("-"*60)
for rank, (_, row) in enumerate(ml_results.head(10).iterrows(), 1):
    key = int(row['MACCS_Key'])
    desc = MACCS_DEFINITIONS.get(key, "Unknown substructure")
    interpretation_lines.append(f"")
    interpretation_lines.append(f"Rank {rank}: MACCS Key {key}")
    interpretation_lines.append(f"  Description: {desc}")
    interpretation_lines.append(f"  RF Importance: {row['RF_Importance']:.6f}")

interpretation_lines.append("")
interpretation_lines.append("CHEMICAL INSIGHTS")
interpretation_lines.append("-"*60)

# Identify patterns in top features
top_positive_keys = top_20[top_20['Correlation'] > 0].head(10)['MACCS_Key'].astype(int).tolist()
top_negative_keys = top_20[top_20['Correlation'] < 0].head(10)['MACCS_Key'].astype(int).tolist()

interpretation_lines.append("")
interpretation_lines.append("Substructures ASSOCIATED WITH POSITIVE Delta_PCE (improvement):")
for key in top_positive_keys[:5]:
    desc = MACCS_DEFINITIONS.get(key, "Unknown")
    interpretation_lines.append(f"  - Key {key}: {desc}")

interpretation_lines.append("")
interpretation_lines.append("Substructures ASSOCIATED WITH NEGATIVE Delta_PCE (degradation):")
for key in top_negative_keys[:5]:
    desc = MACCS_DEFINITIONS.get(key, "Unknown")
    interpretation_lines.append(f"  - Key {key}: {desc}")

interpretation_lines.append("")
interpretation_lines.append("METHODOLOGY NOTES")
interpretation_lines.append("-"*60)
interpretation_lines.append("- MACCS fingerprints contain 166 structural keys (bits)")
interpretation_lines.append("- Point-biserial correlation measures relationship between binary key presence and continuous Delta_PCE")
interpretation_lines.append("- Random Forest and Logistic Regression models use cross-validation for robust performance estimation")
interpretation_lines.append("- ROC-AUC > 0.5 indicates predictive power above random chance")
interpretation_lines.append("- Significance levels: * p<0.05, ** p<0.01, *** p<0.001")
interpretation_lines.append("")
interpretation_lines.append("="*80)
interpretation_lines.append("END OF REPORT")
interpretation_lines.append("="*80)

interpretation_text = "\n".join(interpretation_lines)
interpretation_path = '/share/yhm/test/AutoML_EDA/fingerprints/maccs_interpretation.txt'
with open(interpretation_path, 'w') as f:
    f.write(interpretation_text)
print(f"   Interpretation saved to: {interpretation_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*80)
print(f"\nFiles generated:")
print(f"  1. MACCS Fingerprints: {fp_output_path}")
print(f"  2. Key Correlation: {corr_output_path}")
print(f"  3. ML Results: {ml_output_path}")
print(f"  4. Interpretation: {interpretation_path}")
print(f"\nFigures generated:")
print(f"  1. Top Keys Plot: {top_keys_path}")
print(f"  2. Heatmap: {heatmap_path}")
print(f"  3. PCA Plot: {pca_path}")
print(f"  4. ROC Curves: {roc_path}")
print(f"  5. Feature Importance: {importance_path}")
print("\n" + "="*80)
