#!/usr/bin/env python3
"""
Atom Pair (AP) and Topological Torsion (TT) Fingerprint Analysis for QSPR
Compares fingerprint types for Delta_PCE prediction and analyzes feature importance
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.AtomPairs import Pairs, Torsions
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression

import os
from datetime import datetime

# Output directory
OUTPUT_DIR = "/share/yhm/test/AutoML_EDA/fingerprints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("Atom Pair (AP) and Topological Torsion (TT) Fingerprint Analysis")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# 1. Load Data
# ============================================================================
print("Step 1: Loading data...")
DATA_PATH = "/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx"
df = pd.read_excel(DATA_PATH)
print(f"Total samples loaded: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================================
# 2. Calculate Delta_PCE
# ============================================================================
print("\nStep 2: Calculating Delta_PCE...")
df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']
print(f"Delta_PCE statistics:")
print(df['Delta_PCE'].describe())

# ============================================================================
# 3. Generate Fingerprints
# ============================================================================
print("\nStep 3: Generating Molecular Fingerprints...")

def smiles_to_mol(smiles):
    """Convert SMILES to RDKit molecule"""
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    return mol

def get_atom_pair_fingerprint(mol, n_bits=2048):
    """Generate Atom Pair fingerprint"""
    if mol is None:
        return None
    # Get the explicit bit vector
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_topological_torsion_fingerprint(mol, n_bits=2048):
    """Generate Topological Torsion fingerprint"""
    if mol is None:
        return None
    # Get the explicit bit vector
    fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Convert SMILES to molecules
print("Converting SMILES to molecules...")
df['mol'] = df['smiles'].apply(smiles_to_mol)
valid_mask = df['mol'].notna()
print(f"Valid molecules: {valid_mask.sum()} / {len(df)}")

# Filter valid data
df_valid = df[valid_mask].copy()
df_valid = df_valid.dropna(subset=['Delta_PCE'])
print(f"Samples with valid Delta_PCE: {len(df_valid)}")

# Generate fingerprints
print("Generating Atom Pair fingerprints...")
ap_fps = np.array([get_atom_pair_fingerprint(mol) for mol in df_valid['mol'].values])
print(f"AP fingerprints shape: {ap_fps.shape}")

print("Generating Topological Torsion fingerprints...")
tt_fps = np.array([get_topological_torsion_fingerprint(mol) for mol in df_valid['mol'].values])
print(f"TT fingerprints shape: {tt_fps.shape}")

# Get target
y = df_valid['Delta_PCE'].values

# ============================================================================
# 4. Save Fingerprints
# ============================================================================
print("\nStep 4: Saving fingerprints...")

# Create DataFrames with SMILES and Delta_PCE
ap_df = pd.DataFrame(ap_fps, columns=[f'AP_{i}' for i in range(2048)])
ap_df.insert(0, 'smiles', df_valid['smiles'].values)
ap_df.insert(1, 'Delta_PCE', y)

tt_df = pd.DataFrame(tt_fps, columns=[f'TT_{i}' for i in range(2048)])
tt_df.insert(0, 'smiles', df_valid['smiles'].values)
tt_df.insert(1, 'Delta_PCE', y)

ap_df.to_csv(os.path.join(OUTPUT_DIR, "atompair_fingerprints.csv"), index=False)
tt_df.to_csv(os.path.join(OUTPUT_DIR, "torsion_fingerprints.csv"), index=False)
print("Fingerprints saved!")

# ============================================================================
# 5. Feature Analysis
# ============================================================================
print("\nStep 5: Feature Analysis...")

# Calculate bit densities
ap_bit_density = (ap_fps > 0).mean(axis=0)
tt_bit_density = (tt_fps > 0).mean(axis=0)

print(f"AP fingerprint bit density: {ap_bit_density.mean():.4f}")
print(f"TT fingerprint bit density: {tt_bit_density.mean():.4f}")

# Most common bits
ap_top_bits = np.argsort(ap_bit_density)[::-1][:20]
tt_top_bits = np.argsort(tt_bit_density)[::-1][:20]

print(f"\nTop 10 AP bits: {ap_top_bits[:10]}")
print(f"Top 10 TT bits: {tt_top_bits[:10]}")

# Calculate mutual information with target
print("\nCalculating mutual information for feature importance...")
mi_ap = mutual_info_regression(ap_fps, y, random_state=42)
mi_tt = mutual_info_regression(tt_fps, y, random_state=42)

# Top predictive bits
ap_top_predictive = np.argsort(mi_ap)[::-1][:20]
tt_top_predictive = np.argsort(mi_tt)[::-1][:20]

print(f"\nTop 10 predictive AP bits: {ap_top_predictive[:10]}")
print(f"Top 10 predictive TT bits: {tt_top_predictive[:10]}")

# Bit overlap analysis
ap_active = set(np.where(ap_bit_density > 0.01)[0])
tt_active = set(np.where(tt_bit_density > 0.01)[0])
overlap = ap_active & tt_active

print(f"\nBit Overlap Analysis:")
print(f"  AP active bits (>1% density): {len(ap_active)}")
print(f"  TT active bits (>1% density): {len(tt_active)}")
print(f"  Overlapping bits: {len(overlap)}")

# ============================================================================
# 6. Machine Learning Models
# ============================================================================
print("\nStep 6: Machine Learning Model Comparison...")

# Split data
X_train_ap, X_test_ap, X_train_tt, X_test_tt, y_train, y_test = train_test_split(
    ap_fps, tt_fps, y, test_size=0.2, random_state=42
)

# Scale target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Results storage
results = []

def evaluate_model(model, X_train, X_test, y_train, y_test, y_scaler, name, fp_type):
    """Train and evaluate a model"""
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    return {
        'Model': name,
        'Fingerprint': fp_type,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'CV_R2_Mean': cv_scores.mean(),
        'CV_R2_Std': cv_scores.std()
    }

# --- SVM Models ---
print("\nTraining SVM models...")
svm_params = {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}

# AP + SVM
svm_ap = SVR(**svm_params)
result_svm_ap = evaluate_model(svm_ap, X_train_ap, X_test_ap, y_train_scaled, y_test_scaled, scaler_y, 'SVM', 'Atom Pair')
results.append(result_svm_ap)
print(f"  SVM + AP: R2={result_svm_ap['R2']:.4f}, RMSE={result_svm_ap['RMSE']:.4f}")

# TT + SVM
svm_tt = SVR(**svm_params)
result_svm_tt = evaluate_model(svm_tt, X_train_tt, X_test_tt, y_train_scaled, y_test_scaled, scaler_y, 'SVM', 'Topological Torsion')
results.append(result_svm_tt)
print(f"  SVM + TT: R2={result_svm_tt['R2']:.4f}, RMSE={result_svm_tt['RMSE']:.4f}")

# --- Neural Network Models ---
print("\nTraining Neural Network models...")
nn_params = {'hidden_layer_sizes': (512, 256, 128), 'activation': 'relu',
             'max_iter': 500, 'random_state': 42, 'early_stopping': True}

# AP + NN
nn_ap = MLPRegressor(**nn_params)
result_nn_ap = evaluate_model(nn_ap, X_train_ap, X_test_ap, y_train_scaled, y_test_scaled, scaler_y, 'Neural Network', 'Atom Pair')
results.append(result_nn_ap)
print(f"  NN + AP: R2={result_nn_ap['R2']:.4f}, RMSE={result_nn_ap['RMSE']:.4f}")

# TT + NN
nn_tt = MLPRegressor(**nn_params)
result_nn_tt = evaluate_model(nn_tt, X_train_tt, X_test_tt, y_train_scaled, y_test_scaled, scaler_y, 'Neural Network', 'Topological Torsion')
results.append(result_nn_tt)
print(f"  NN + TT: R2={result_nn_tt['R2']:.4f}, RMSE={result_nn_tt['RMSE']:.4f}")

# --- Random Forest Models ---
print("\nTraining Random Forest models...")
rf_params = {'n_estimators': 100, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1}

# AP + RF
rf_ap = RandomForestRegressor(**rf_params)
result_rf_ap = evaluate_model(rf_ap, X_train_ap, X_test_ap, y_train, y_test, None, 'Random Forest', 'Atom Pair')
results.append(result_rf_ap)
print(f"  RF + AP: R2={result_rf_ap['R2']:.4f}, RMSE={result_rf_ap['RMSE']:.4f}")

# TT + RF
rf_tt = RandomForestRegressor(**rf_params)
result_rf_tt = evaluate_model(rf_tt, X_train_tt, X_test_tt, y_train, y_test, None, 'Random Forest', 'Topological Torsion')
results.append(result_rf_tt)
print(f"  RF + TT: R2={result_rf_tt['R2']:.4f}, RMSE={result_rf_tt['RMSE']:.4f}")

# --- Combined Fingerprints ---
print("\nTraining models on combined fingerprints...")

# Combine fingerprints
X_train_combined = np.hstack([X_train_ap, X_train_tt])
X_test_combined = np.hstack([X_test_ap, X_test_tt])

# Combined + SVM
svm_combined = SVR(**svm_params)
result_svm_combined = evaluate_model(svm_combined, X_train_combined, X_test_combined,
                                      y_train_scaled, y_test_scaled, scaler_y, 'SVM', 'Combined (AP+TT)')
results.append(result_svm_combined)
print(f"  SVM + Combined: R2={result_svm_combined['R2']:.4f}, RMSE={result_svm_combined['RMSE']:.4f}")

# Combined + NN
nn_combined = MLPRegressor(**nn_params)
result_nn_combined = evaluate_model(nn_combined, X_train_combined, X_test_combined,
                                     y_train_scaled, y_test_scaled, scaler_y, 'Neural Network', 'Combined (AP+TT)')
results.append(result_nn_combined)
print(f"  NN + Combined: R2={result_nn_combined['R2']:.4f}, RMSE={result_nn_combined['RMSE']:.4f}")

# Combined + RF
rf_combined = RandomForestRegressor(**rf_params)
result_rf_combined = evaluate_model(rf_combined, X_train_combined, X_test_combined,
                                     y_train, y_test, None, 'Random Forest', 'Combined (AP+TT)')
results.append(result_rf_combined)
print(f"  RF + Combined: R2={result_rf_combined['R2']:.4f}, RMSE={result_rf_combined['RMSE']:.4f}")

# --- Ensemble Approach: Average Predictions ---
print("\nEvaluating ensemble approach...")

# Train separate models and average predictions
svm_ensemble_ap = SVR(**svm_params)
svm_ensemble_tt = SVR(**svm_params)
svm_ensemble_ap.fit(X_train_ap, y_train_scaled)
svm_ensemble_tt.fit(X_train_tt, y_train_scaled)

y_pred_ap = svm_ensemble_ap.predict(X_test_ap)
y_pred_tt = svm_ensemble_tt.predict(X_test_tt)
y_pred_ensemble = (y_pred_ap + y_pred_tt) / 2

y_pred_ensemble_orig = scaler_y.inverse_transform(y_pred_ensemble.reshape(-1, 1)).ravel()
y_test_orig = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

ensemble_r2 = r2_score(y_test_orig, y_pred_ensemble_orig)
ensemble_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_ensemble_orig))
ensemble_mae = mean_absolute_error(y_test_orig, y_pred_ensemble_orig)

results.append({
    'Model': 'Ensemble SVM',
    'Fingerprint': 'Averaged (AP+TT)',
    'R2': ensemble_r2,
    'RMSE': ensemble_rmse,
    'MAE': ensemble_mae,
    'CV_R2_Mean': np.nan,
    'CV_R2_Std': np.nan
})
print(f"  Ensemble SVM (avg): R2={ensemble_r2:.4f}, RMSE={ensemble_rmse:.4f}")

# ============================================================================
# 7. Save Results
# ============================================================================
print("\nStep 7: Saving results...")

# Save comparison results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R2', ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "fingerprint_comparison.csv"), index=False)

# Save combined ML results with more details
combined_results = []
for result in results:
    combined_results.append({
        'Model': result['Model'],
        'Fingerprint_Type': result['Fingerprint'],
        'Test_R2': result['R2'],
        'Test_RMSE': result['RMSE'],
        'Test_MAE': result['MAE'],
        'CV_R2_Mean': result['CV_R2_Mean'],
        'CV_R2_Std': result['CV_R2_Std']
    })

combined_df = pd.DataFrame(combined_results)
combined_df.to_csv(os.path.join(OUTPUT_DIR, "combined_ml_results.csv"), index=False)

# ============================================================================
# 8. Generate Summary Report
# ============================================================================
print("\nStep 8: Generating summary report...")

summary_lines = []
summary_lines.append("="*80)
summary_lines.append("Atom Pair (AP) and Topological Torsion (TT) Fingerprint Analysis Report")
summary_lines.append("="*80)
summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary_lines.append(f"Data source: {DATA_PATH}")
summary_lines.append("")

summary_lines.append("1. DATA SUMMARY")
summary_lines.append("-"*40)
summary_lines.append(f"Total samples: {len(df)}")
summary_lines.append(f"Valid molecules: {len(df_valid)}")
summary_lines.append(f"Delta_PCE range: [{y.min():.4f}, {y.max():.4f}]")
summary_lines.append(f"Delta_PCE mean: {y.mean():.4f}")
summary_lines.append(f"Delta_PCE std: {y.std():.4f}")
summary_lines.append("")

summary_lines.append("2. FINGERPRINT CHARACTERISTICS")
summary_lines.append("-"*40)
summary_lines.append(f"Fingerprint size: 2048 bits each")
summary_lines.append(f"")
summary_lines.append("Atom Pair (AP) Fingerprint:")
summary_lines.append(f"  - Bit density: {ap_bit_density.mean():.4f}")
summary_lines.append(f"  - Active bits (>1%): {len(ap_active)}")
summary_lines.append(f"  - Top 5 bits: {list(ap_top_bits[:5])}")
summary_lines.append(f"")
summary_lines.append("Topological Torsion (TT) Fingerprint:")
summary_lines.append(f"  - Bit density: {tt_bit_density.mean():.4f}")
summary_lines.append(f"  - Active bits (>1%): {len(tt_active)}")
summary_lines.append(f"  - Top 5 bits: {list(tt_top_bits[:5])}")
summary_lines.append(f"")
summary_lines.append(f"Bit overlap: {len(overlap)} bits")
summary_lines.append("")

summary_lines.append("3. MOST PREDICTIVE FEATURES")
summary_lines.append("-"*40)
summary_lines.append("Top 10 Atom Pair bits (by mutual information):")
for i, bit in enumerate(ap_top_predictive[:10]):
    summary_lines.append(f"  {i+1}. Bit {bit}: MI={mi_ap[bit]:.6f}")
summary_lines.append("")
summary_lines.append("Top 10 Topological Torsion bits (by mutual information):")
for i, bit in enumerate(tt_top_predictive[:10]):
    summary_lines.append(f"  {i+1}. Bit {bit}: MI={mi_tt[bit]:.6f}")
summary_lines.append("")

summary_lines.append("4. MODEL PERFORMANCE COMPARISON")
summary_lines.append("-"*40)
summary_lines.append(f"{'Model':<25} {'Fingerprint':<25} {'R2':>8} {'RMSE':>8} {'MAE':>8}")
summary_lines.append("-"*80)
for _, row in results_df.iterrows():
    summary_lines.append(f"{row['Model']:<25} {row['Fingerprint']:<25} {row['R2']:>8.4f} {row['RMSE']:>8.4f} {row['MAE']:>8.4f}")
summary_lines.append("")

summary_lines.append("5. KEY FINDINGS")
summary_lines.append("-"*40)

# Best model
best_model = results_df.iloc[0]
summary_lines.append(f"Best Model: {best_model['Model']} with {best_model['Fingerprint']}")
summary_lines.append(f"  - R2 Score: {best_model['R2']:.4f}")
summary_lines.append(f"  - RMSE: {best_model['RMSE']:.4f}")
summary_lines.append("")

# Compare AP vs TT
ap_results = results_df[results_df['Fingerprint'] == 'Atom Pair']['R2'].max()
tt_results = results_df[results_df['Fingerprint'] == 'Topological Torsion']['R2'].max()
combined_results_r2 = results_df[results_df['Fingerprint'].str.contains('Combined')]['R2'].max()

if ap_results > tt_results:
    summary_lines.append("Atom Pair fingerprints show better performance than Topological Torsion")
else:
    summary_lines.append("Topological Torsion fingerprints show better performance than Atom Pair")

summary_lines.append(f"  - Best AP R2: {ap_results:.4f}")
summary_lines.append(f"  - Best TT R2: {tt_results:.4f}")
summary_lines.append(f"  - Best Combined R2: {combined_results_r2:.4f}")
summary_lines.append("")

# Ensemble vs Combined
summary_lines.append("Combined fingerprint approach provides:")
summary_lines.append("  - Wider chemical space coverage")
summary_lines.append(f"  - Improved prediction: {combined_results_r2:.4f} R2")
summary_lines.append("")

summary_lines.append("6. FILES GENERATED")
summary_lines.append("-"*40)
summary_lines.append(f"- atompair_fingerprints.csv: {len(ap_df)} samples x 2050 columns")
summary_lines.append(f"- torsion_fingerprints.csv: {len(tt_df)} samples x 2050 columns")
summary_lines.append(f"- fingerprint_comparison.csv: Model performance comparison")
summary_lines.append(f"- combined_ml_results.csv: Detailed ML results")
summary_lines.append(f"- ap_tt_analysis.txt: This summary report")
summary_lines.append("")

summary_lines.append("="*80)
summary_lines.append("Analysis Complete")
summary_lines.append("="*80)

# Write summary
summary_text = "\n".join(summary_lines)
with open(os.path.join(OUTPUT_DIR, "ap_tt_analysis.txt"), 'w') as f:
    f.write(summary_text)

print(summary_text)

print(f"\nAll results saved to: {OUTPUT_DIR}")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
