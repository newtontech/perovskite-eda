#!/usr/bin/env python3
"""
QSPR Correlation Analysis for Perovskite Solar Cell Modulators
This script performs correlation analysis between chemical descriptors and Delta_PCE.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# File paths
INPUT_FILE = "/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx"
OUTPUT_DIR = "/share/yhm/test/AutoML_EDA"
PROCESSED_DATA = os.path.join(OUTPUT_DIR, "processed_data.csv")
CORRELATION_MATRIX = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
HEATMAP_FIG = os.path.join(OUTPUT_DIR, "figures/correlation_heatmap.png")
ANALYSIS_TEXT = os.path.join(OUTPUT_DIR, "correlation_analysis.txt")

# Chemical features for analysis
CHEMICAL_FEATURES = ['molecular_weight', 'h_bond_donors', 'h_bond_acceptors',
                     'rotatable_bonds', 'tpsa', 'log_p']

def load_and_process_data():
    """Load data from Excel file or processed CSV, and prepare for analysis."""
    print("=" * 70)
    print("QSPR CORRELATION ANALYSIS FOR PEROVSKITE SOLAR CELL MODULATORS")
    print("=" * 70)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if processed data already exists
    if os.path.exists(PROCESSED_DATA):
        print(f"\nLoading processed data from {PROCESSED_DATA}...")
        df = pd.read_csv(PROCESSED_DATA)
        print(f"Loaded {len(df)} samples")
        return df

    # Otherwise, load from Excel and process
    print(f"\nLoading data from Excel file...")
    print("This may take a few minutes for large files...")

    # Load data with openpyxl engine
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    initial_samples = len(df)
    print(f"Loaded {initial_samples} samples with {len(df.columns)} columns")

    # Calculate Delta_PCE
    print("\nCalculating Delta_PCE...")
    df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']
    print(f"Delta_PCE range: [{df['Delta_PCE'].min():.4f}, {df['Delta_PCE'].max():.4f}]")

    # Select relevant columns
    identifier_cols = ['cas_number', 'pubchem_id', 'smiles', 'molecular_formula']
    analysis_cols = CHEMICAL_FEATURES + ['Delta_PCE']

    selected_cols = []
    for col in identifier_cols + analysis_cols:
        if col in df.columns:
            selected_cols.append(col)

    df_selected = df[selected_cols].copy()

    # Remove rows with missing values in analysis columns
    df_clean = df_selected.dropna(subset=analysis_cols).copy()
    final_samples = len(df_clean)
    print(f"Removed {initial_samples - final_samples} samples with missing values")
    print(f"Retained {final_samples} complete samples for analysis")

    # Save processed data for future use
    df_clean.to_csv(PROCESSED_DATA, index=False)
    print(f"Saved processed data to {PROCESSED_DATA}")

    return df_clean

def calculate_pearson_correlations(df, features, target='Delta_PCE'):
    """Calculate Pearson correlations between features and target."""
    correlations = {}
    p_values = {}

    for feature in features:
        if feature in df.columns:
            valid_data = df[[feature, target]].dropna()
            if len(valid_data) > 2:
                corr, pval = pearsonr(valid_data[feature], valid_data[target])
                correlations[feature] = corr
                p_values[feature] = pval

    return correlations, p_values

def calculate_spearman_correlations(df, features, target='Delta_PCE'):
    """Calculate Spearman correlations between features and target."""
    correlations = {}
    p_values = {}

    for feature in features:
        if feature in df.columns:
            valid_data = df[[feature, target]].dropna()
            if len(valid_data) > 2:
                corr, pval = spearmanr(valid_data[feature], valid_data[target])
                correlations[feature] = corr
                p_values[feature] = pval

    return correlations, p_values

def calculate_partial_correlation(df, x, y, control_vars):
    """
    Calculate partial correlation between x and y controlling for control_vars.
    Uses the formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    For multiple control variables, uses residuals method.
    """
    from sklearn.linear_model import LinearRegression

    # Get complete cases
    all_vars = [x, y] + control_vars
    valid_data = df[all_vars].dropna()

    if len(valid_data) < len(all_vars) + 2:
        return np.nan, np.nan

    # Regress x on control variables
    lr_x = LinearRegression()
    lr_x.fit(valid_data[control_vars], valid_data[x])
    residual_x = valid_data[x] - lr_x.predict(valid_data[control_vars])

    # Regress y on control variables
    lr_y = LinearRegression()
    lr_y.fit(valid_data[control_vars], valid_data[y])
    residual_y = valid_data[y] - lr_y.predict(valid_data[control_vars])

    # Correlation of residuals
    corr, pval = pearsonr(residual_x, residual_y)
    return corr, pval

def create_correlation_heatmap(corr_matrix, output_path):
    """Create and save correlation heatmap."""
    plt.figure(figsize=(10, 8))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Create heatmap
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"})

    plt.title('Correlation Matrix: Chemical Descriptors and Delta_PCE', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {output_path}")

def main():
    # Step 1: Load and process data
    df = load_and_process_data()

    # Verify we have the required columns
    features_present = [f for f in CHEMICAL_FEATURES if f in df.columns]
    print(f"\nChemical features available for analysis: {features_present}")

    if 'Delta_PCE' not in df.columns:
        print("ERROR: Delta_PCE column not found!")
        return

    # Step 2: Calculate Pearson correlations
    print("\n" + "=" * 70)
    print("PEARSON CORRELATION ANALYSIS")
    print("=" * 70)

    pearson_corr, pearson_pval = calculate_pearson_correlations(df, features_present, 'Delta_PCE')

    print("\nPearson Correlations with Delta_PCE:")
    print("-" * 50)
    for feature in sorted(pearson_corr.keys(), key=lambda x: abs(pearson_corr[x]), reverse=True):
        sig = "***" if pearson_pval[feature] < 0.001 else "**" if pearson_pval[feature] < 0.01 else "*" if pearson_pval[feature] < 0.05 else ""
        print(f"  {feature:25s}: r = {pearson_corr[feature]:7.4f}, p = {pearson_pval[feature]:.4e} {sig}")
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")

    # Step 3: Calculate Spearman correlations
    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATION ANALYSIS")
    print("=" * 70)

    spearman_corr, spearman_pval = calculate_spearman_correlations(df, features_present, 'Delta_PCE')

    print("\nSpearman Correlations with Delta_PCE:")
    print("-" * 50)
    for feature in sorted(spearman_corr.keys(), key=lambda x: abs(spearman_corr[x]), reverse=True):
        sig = "***" if spearman_pval[feature] < 0.001 else "**" if spearman_pval[feature] < 0.01 else "*" if spearman_pval[feature] < 0.05 else ""
        print(f"  {feature:25s}: rho = {spearman_corr[feature]:7.4f}, p = {spearman_pval[feature]:.4e} {sig}")

    # Step 4: Create correlation matrix (including inter-feature correlations)
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX")
    print("=" * 70)

    analysis_cols = features_present + ['Delta_PCE']
    correlation_matrix = df[analysis_cols].corr(method='pearson')

    print("\nFull Correlation Matrix:")
    print(correlation_matrix.round(4).to_string())

    # Save correlation matrix
    correlation_matrix.to_csv(CORRELATION_MATRIX)
    print(f"\nSaved correlation matrix to {CORRELATION_MATRIX}")

    # Step 5: Create heatmap
    print("\n" + "=" * 70)
    print("CREATING CORRELATION HEATMAP")
    print("=" * 70)

    # Ensure figures directory exists
    os.makedirs(os.path.dirname(HEATMAP_FIG), exist_ok=True)
    create_correlation_heatmap(correlation_matrix, HEATMAP_FIG)

    # Step 6: Partial correlation analysis (controlling for molecular_weight)
    print("\n" + "=" * 70)
    print("PARTIAL CORRELATION ANALYSIS")
    print("(Controlling for molecular_weight)")
    print("=" * 70)

    partial_corrs = {}
    control_var = 'molecular_weight'

    if control_var in df.columns:
        print("\nPartial Correlations (controlling for molecular_weight):")
        print("-" * 60)

        for feature in features_present:
            if feature != control_var:
                pcorr, ppval = calculate_partial_correlation(df, feature, 'Delta_PCE', [control_var])
                if not np.isnan(pcorr):
                    partial_corrs[feature] = (pcorr, ppval)
                    sig = "***" if ppval < 0.001 else "**" if ppval < 0.01 else "*" if ppval < 0.05 else ""
                    print(f"  {feature:25s}: r_partial = {pcorr:7.4f}, p = {ppval:.4e} {sig}")
    else:
        print(f"Warning: {control_var} not found in data, skipping partial correlation analysis")

    # Step 7: Identify top 5 most correlated features
    print("\n" + "=" * 70)
    print("TOP 5 MOST CORRELATED FEATURES WITH Delta_PCE")
    print("=" * 70)

    # Combine Pearson and Spearman results
    all_features_stats = {}
    for feature in features_present:
        all_features_stats[feature] = {
            'pearson_r': pearson_corr.get(feature, np.nan),
            'pearson_p': pearson_pval.get(feature, np.nan),
            'spearman_rho': spearman_corr.get(feature, np.nan),
            'spearman_p': spearman_pval.get(feature, np.nan),
            'partial_r': partial_corrs.get(feature, (np.nan, np.nan))[0] if feature in partial_corrs else np.nan,
            'partial_p': partial_corrs.get(feature, (np.nan, np.nan))[1] if feature in partial_corrs else np.nan
        }

    # Rank by absolute Pearson correlation
    ranked_features = sorted(features_present,
                            key=lambda x: abs(pearson_corr.get(x, 0)),
                            reverse=True)

    print("\nTop 5 Features (ranked by absolute Pearson correlation):")
    print("-" * 80)
    print(f"{'Rank':<6}{'Feature':<25}{'Pearson r':<12}{'Spearman rho':<14}{'Partial r':<12}{'Best':<8}")
    print("-" * 80)

    for i, feature in enumerate(ranked_features[:5], 1):
        stats = all_features_stats[feature]
        pearson_abs = abs(stats['pearson_r'])
        spearman_abs = abs(stats['spearman_rho'])
        partial_abs = abs(stats['partial_r']) if not np.isnan(stats['partial_r']) else 0

        # Determine which correlation is strongest
        max_corr = max(pearson_abs, spearman_abs)
        best = 'Pearson' if pearson_abs >= spearman_abs else 'Spearman'

        print(f"{i:<6}{feature:<25}{stats['pearson_r']:>10.4f}  {stats['spearman_rho']:>12.4f}  {stats['partial_r']:>10.4f}  {best:<8}")

    # Step 8: Save comprehensive analysis report
    print("\n" + "=" * 70)
    print("SAVING ANALYSIS REPORT")
    print("=" * 70)

    with open(ANALYSIS_TEXT, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("QSPR CORRELATION ANALYSIS REPORT FOR PEROVSKITE SOLAR CELL MODULATORS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples analyzed: {len(df)}\n")
        f.write(f"Chemical features analyzed: {len(features_present)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("1. PEARSON CORRELATIONS WITH Delta_PCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Feature':<25}{'r':<12}{'p-value':<15}{'Significance':<12}\n")
        f.write("-" * 80 + "\n")
        for feature in sorted(pearson_corr.keys(), key=lambda x: abs(pearson_corr[x]), reverse=True):
            sig = "***" if pearson_pval[feature] < 0.001 else "**" if pearson_pval[feature] < 0.01 else "*" if pearson_pval[feature] < 0.05 else "ns"
            f.write(f"{feature:<25}{pearson_corr[feature]:>10.4f}  {pearson_pval[feature]:>13.4e}  {sig:<12}\n")
        f.write("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n\n")

        f.write("-" * 80 + "\n")
        f.write("2. SPEARMAN CORRELATIONS WITH Delta_PCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Feature':<25}{'rho':<12}{'p-value':<15}{'Significance':<12}\n")
        f.write("-" * 80 + "\n")
        for feature in sorted(spearman_corr.keys(), key=lambda x: abs(spearman_corr[x]), reverse=True):
            sig = "***" if spearman_pval[feature] < 0.001 else "**" if spearman_pval[feature] < 0.01 else "*" if spearman_pval[feature] < 0.05 else "ns"
            f.write(f"{feature:<25}{spearman_corr[feature]:>10.4f}  {spearman_pval[feature]:>13.4e}  {sig:<12}\n")
        f.write("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n\n")

        f.write("-" * 80 + "\n")
        f.write("3. PARTIAL CORRELATIONS (Controlling for molecular_weight)\n")
        f.write("-" * 80 + "\n")
        if partial_corrs:
            f.write(f"{'Feature':<25}{'r_partial':<12}{'p-value':<15}{'Significance':<12}\n")
            f.write("-" * 80 + "\n")
            for feature in sorted(partial_corrs.keys(), key=lambda x: abs(partial_corrs[x][0]), reverse=True):
                pcorr, ppval = partial_corrs[feature]
                sig = "***" if ppval < 0.001 else "**" if ppval < 0.01 else "*" if ppval < 0.05 else "ns"
                f.write(f"{feature:<25}{pcorr:>10.4f}  {ppval:>13.4e}  {sig:<12}\n")
            f.write("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n\n")
        else:
            f.write("Partial correlation analysis not performed (molecular_weight not available)\n\n")

        f.write("-" * 80 + "\n")
        f.write("4. TOP 5 MOST CORRELATED FEATURES\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6}{'Feature':<25}{'Pearson r':<14}{'Spearman rho':<14}{'Partial r':<12}\n")
        f.write("-" * 80 + "\n")
        for i, feature in enumerate(ranked_features[:5], 1):
            stats = all_features_stats[feature]
            partial_str = f"{stats['partial_r']:.4f}" if not np.isnan(stats['partial_r']) else "N/A"
            f.write(f"{i:<6}{feature:<25}{stats['pearson_r']:>12.4f}  {stats['spearman_rho']:>12.4f}  {partial_str:>10}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("5. KEY FINDINGS AND INTERPRETATION\n")
        f.write("-" * 80 + "\n")

        # Identify significant correlations
        sig_pearson = [(f, pearson_corr[f], pearson_pval[f]) for f in pearson_corr if pearson_pval[f] < 0.05]
        sig_spearman = [(f, spearman_corr[f], spearman_pval[f]) for f in spearman_corr if spearman_pval[f] < 0.05]

        f.write(f"\nSignificant Pearson correlations (p < 0.05): {len(sig_pearson)}\n")
        f.write(f"Significant Spearman correlations (p < 0.05): {len(sig_spearman)}\n\n")

        if ranked_features:
            top_feature = ranked_features[0]
            f.write(f"Strongest correlation: {top_feature}\n")
            f.write(f"  Pearson r = {pearson_corr[top_feature]:.4f} (p = {pearson_pval[top_feature]:.4e})\n")
            f.write(f"  Spearman rho = {spearman_corr[top_feature]:.4f} (p = {spearman_pval[top_feature]:.4e})\n\n")

        f.write("Correlation interpretation guidelines:\n")
        f.write("  |r| < 0.1: negligible correlation\n")
        f.write("  0.1 <= |r| < 0.3: weak correlation\n")
        f.write("  0.3 <= |r| < 0.5: moderate correlation\n")
        f.write("  0.5 <= |r| < 0.7: strong correlation\n")
        f.write("  |r| >= 0.7: very strong correlation\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("=" * 80 + "\n")
        f.write(f"1. Processed data: {PROCESSED_DATA}\n")
        f.write(f"2. Correlation matrix: {CORRELATION_MATRIX}\n")
        f.write(f"3. Correlation heatmap: {HEATMAP_FIG}\n")
        f.write(f"4. This report: {ANALYSIS_TEXT}\n")
        f.write("=" * 80 + "\n")

    print(f"Saved analysis report to {ANALYSIS_TEXT}")

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput files generated:")
    print(f"  1. Processed data: {PROCESSED_DATA}")
    print(f"  2. Correlation matrix: {CORRELATION_MATRIX}")
    print(f"  3. Correlation heatmap: {HEATMAP_FIG}")
    print(f"  4. Analysis report: {ANALYSIS_TEXT}")

    return correlation_matrix, pearson_corr, spearman_corr

if __name__ == "__main__":
    main()
