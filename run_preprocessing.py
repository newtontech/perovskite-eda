#!/usr/bin/env python3
"""
QSPR Data Preprocessing for Perovskite Solar Cell Analysis
Fixed version with proper type handling
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# File paths
INPUT_FILE = "/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx"
OUTPUT_DATA = "/share/yhm/test/AutoML_EDA/processed_data.csv"
OUTPUT_SUMMARY = "/share/yhm/test/AutoML_EDA/data_summary.txt"

def main():
    print("=" * 60)
    print("QSPR Data Preprocessing for Perovskite Solar Cell Analysis")
    print("=" * 60)

    # Step 1: Load the data
    print("\n[Step 1] Loading data from Excel file...")
    df = pd.read_excel(INPUT_FILE)
    initial_samples = len(df)
    print(f"Loaded {initial_samples} samples with {len(df.columns)} columns")

    # Step 2: Convert PCE columns to numeric
    print("\n[Step 2] Converting PCE columns to numeric...")
    pce_cols = ['jv_reverse_scan_pce', 'jv_reverse_scan_pce_without_modulator']
    for col in pce_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"  {col}: {df[col].notna().sum()} valid numeric values")

    # Step 3: Calculate Delta_PCE
    print("\n[Step 3] Calculating Delta_PCE...")
    df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']
    valid_delta = df['Delta_PCE'].notna().sum()
    print(f"Delta_PCE calculated. Valid values: {valid_delta}")
    print(f"Range: [{df['Delta_PCE'].min():.4f}, {df['Delta_PCE'].max():.4f}]")

    # Step 4: Select relevant columns for QSPR analysis
    print("\n[Step 4] Selecting relevant columns for QSPR analysis...")
    chemical_features = ['molecular_weight', 'h_bond_donors', 'h_bond_acceptors',
                         'rotatable_bonds', 'tpsa', 'log_p']
    target = ['Delta_PCE']
    identifier_cols = ['cas_number', 'pubchem_id', 'smiles', 'molecular_formula']

    # Also include other useful columns
    other_cols = ['jv_reverse_scan_pce', 'jv_reverse_scan_pce_without_modulator',
                  'jv_reverse_scan_j_sc', 'jv_reverse_scan_v_oc', 'jv_reverse_scan_ff',
                  'interfacial_modulator_material_type']

    selected_cols = []
    for col in identifier_cols + chemical_features + other_cols + target:
        if col in df.columns:
            selected_cols.append(col)

    df_selected = df[selected_cols].copy()
    print(f"Selected {len(selected_cols)} columns")

    # Step 5: Missing value analysis before cleaning
    print("\n[Step 5] Analyzing missing values...")
    missing_before = df_selected.isnull().sum()
    print("\nMissing values before cleaning:")
    for col in df_selected.columns:
        missing_count = missing_before[col]
        missing_pct = (missing_count / len(df_selected)) * 100
        print(f"  {col}: {missing_count} ({missing_pct:.2f}%)")

    # Step 6: Handle missing values - need complete cases for QSPR
    print("\n[Step 6] Handling missing values...")
    analysis_cols = chemical_features + target
    df_clean = df_selected.dropna(subset=analysis_cols).copy()
    final_samples = len(df_clean)

    print(f"Removed {initial_samples - final_samples} samples with missing values")
    print(f"Retained {final_samples} complete samples ({100*final_samples/initial_samples:.2f}%)")

    # Step 7: Generate statistics
    print("\n[Step 7] Generating statistics...")

    # Chemical features statistics
    print("\nChemical Features Statistics:")
    chem_stats = df_clean[chemical_features].describe()
    print(chem_stats.round(4).to_string())

    # Delta_PCE statistics
    print("\nDelta_PCE Statistics:")
    pce_stats = df_clean['Delta_PCE'].describe()
    print(pce_stats.round(4).to_string())

    # Additional Delta_PCE statistics
    print(f"\nAdditional Delta_PCE Analysis:")
    print(f"  Median: {df_clean['Delta_PCE'].median():.4f}")
    print(f"  Skewness: {df_clean['Delta_PCE'].skew():.4f}")
    print(f"  Kurtosis: {df_clean['Delta_PCE'].kurtosis():.4f}")
    print(f"  Positive Delta_PCE count: {(df_clean['Delta_PCE'] > 0).sum()}")
    print(f"  Negative Delta_PCE count: {(df_clean['Delta_PCE'] < 0).sum()}")
    print(f"  Zero Delta_PCE count: {(df_clean['Delta_PCE'] == 0).sum()}")

    # Step 8: Save processed data
    print(f"\n[Step 8] Saving processed data to {OUTPUT_DATA}...")
    df_clean.to_csv(OUTPUT_DATA, index=False)
    print(f"Saved {len(df_clean)} samples to {OUTPUT_DATA}")

    # Step 9: Save summary report
    print(f"\n[Step 9] Saving summary report to {OUTPUT_SUMMARY}...")

    with open(OUTPUT_SUMMARY, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("QSPR DATA SUMMARY REPORT FOR PEROVSKITE SOLAR CELL ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source file: {INPUT_FILE}\n\n")

        f.write("-" * 70 + "\n")
        f.write("SAMPLE COUNTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Samples before cleaning: {initial_samples}\n")
        f.write(f"Samples after cleaning: {final_samples}\n")
        f.write(f"Samples removed: {initial_samples - final_samples} ({(initial_samples - final_samples)/initial_samples*100:.2f}%)\n\n")

        f.write("-" * 70 + "\n")
        f.write("MISSING VALUE ANALYSIS (BEFORE CLEANING)\n")
        f.write("-" * 70 + "\n")
        for col in df_selected.columns:
            missing_count = missing_before[col]
            missing_pct = (missing_count / initial_samples) * 100
            f.write(f"{col:40s}: {missing_count:6d} ({missing_pct:6.2f}%)\n")
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("CHEMICAL FEATURES STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(chem_stats.round(4).to_string())
        f.write("\n\n")

        f.write("-" * 70 + "\n")
        f.write("TARGET VARIABLE: Delta_PCE STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Delta_PCE = jv_reverse_scan_pce - jv_reverse_scan_pce_without_modulator\n\n")
        f.write(pce_stats.round(4).to_string())
        f.write("\n\n")

        f.write("Additional Delta_PCE Analysis:\n")
        f.write(f"  Median: {df_clean['Delta_PCE'].median():.4f}\n")
        f.write(f"  Skewness: {df_clean['Delta_PCE'].skew():.4f}\n")
        f.write(f"  Kurtosis: {df_clean['Delta_PCE'].kurtosis():.4f}\n")
        f.write(f"  Positive Delta_PCE count: {(df_clean['Delta_PCE'] > 0).sum()}\n")
        f.write(f"  Negative Delta_PCE count: {(df_clean['Delta_PCE'] < 0).sum()}\n")
        f.write(f"  Zero Delta_PCE count: {(df_clean['Delta_PCE'] == 0).sum()}\n\n")

        f.write("-" * 70 + "\n")
        f.write("CORRELATION MATRIX (Chemical Features vs Delta_PCE)\n")
        f.write("-" * 70 + "\n")
        corr_with_target = df_clean[chemical_features + ['Delta_PCE']].corr()['Delta_PCE'].drop('Delta_PCE')
        f.write(corr_with_target.round(4).to_string())
        f.write("\n\n")

        f.write("-" * 70 + "\n")
        f.write("SELECTED COLUMNS FOR QSPR ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write("Identifiers:\n")
        for col in identifier_cols:
            if col in df_clean.columns:
                f.write(f"  - {col}\n")
        f.write("\nChemical Features:\n")
        for col in chemical_features:
            f.write(f"  - {col}\n")
        f.write("\nTarget Variable:\n")
        f.write("  - Delta_PCE\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"Summary report saved to {OUTPUT_SUMMARY}")

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return df_clean

if __name__ == "__main__":
    df = main()
