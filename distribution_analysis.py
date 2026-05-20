#!/usr/bin/env python3
"""
Distribution Analysis for QSPR Data on Perovskite Solar Cells
This script analyzes the distribution of Delta_PCE and chemical descriptors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, skew, kurtosis
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Paths
DATA_PATH = '/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx'
OUTPUT_DIR = '/share/yhm/test/AutoML_EDA'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, 'processed_data.csv')

# Chemical descriptor columns
CHEMICAL_DESCRIPTORS = [
    'molecular_weight', 'h_bond_donors', 'h_bond_acceptors',
    'rotatable_bonds', 'tpsa', 'log_p'
]

# JV columns
JV_COLUMNS = [
    'jv_reverse_scan_pce_without_modulator', 'jv_reverse_scan_j_sc_without_modulator',
    'jv_reverse_scan_v_oc_without_modulator', 'jv_reverse_scan_ff_without_modulator',
    'jv_reverse_scan_pce', 'jv_reverse_scan_j_sc',
    'jv_reverse_scan_v_oc', 'jv_reverse_scan_ff',
    'jv_hysteresis_index_without_modulator', 'jv_hysteresis_index'
]

def load_and_preprocess_data():
    """Load and preprocess the data."""
    print("=" * 60)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 60)

    # Load data
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    print(f"Original data shape: {df.shape}")

    # Calculate Delta_PCE
    print("\nCalculating Delta_PCE...")
    df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']

    # Select relevant columns
    columns_to_keep = CHEMICAL_DESCRIPTORS + ['Delta_PCE']
    df_analysis = df[columns_to_keep].copy()

    # Remove rows with missing values
    print(f"Rows before removing NaN: {len(df_analysis)}")
    df_analysis = df_analysis.dropna()
    print(f"Rows after removing NaN: {len(df_analysis)}")

    # Remove infinite values
    df_analysis = df_analysis.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Rows after removing Inf: {len(df_analysis)}")

    # Save processed data
    df_analysis.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nProcessed data saved to: {PROCESSED_DATA_PATH}")

    return df_analysis

def analyze_delta_pce_distribution(df):
    """Analyze the distribution of Delta_PCE."""
    print("\n" + "=" * 60)
    print("STEP 2: Analyzing Delta_PCE Distribution")
    print("=" * 60)

    delta_pce = df['Delta_PCE'].dropna()

    # Basic statistics
    stats_summary = {
        'Count': len(delta_pce),
        'Mean': delta_pce.mean(),
        'Median': delta_pce.median(),
        'Std': delta_pce.std(),
        'Min': delta_pce.min(),
        'Max': delta_pce.max(),
        'Range': delta_pce.max() - delta_pce.min(),
        'Skewness': skew(delta_pce),
        'Kurtosis': kurtosis(delta_pce),
        'IQR': delta_pce.quantile(0.75) - delta_pce.quantile(0.25)
    }

    print("\nBasic Statistics for Delta_PCE:")
    for key, value in stats_summary.items():
        print(f"  {key}: {value:.4f}")

    # Normality tests
    print("\nNormality Tests:")

    # Shapiro-Wilk test (for smaller samples, use subset if too large)
    sample_size = min(5000, len(delta_pce))
    sample_data = delta_pce.sample(n=sample_size, random_state=42) if len(delta_pce) > sample_size else delta_pce

    shapiro_stat, shapiro_p = shapiro(sample_data)
    print(f"  Shapiro-Wilk Test:")
    print(f"    Statistic: {shapiro_stat:.4f}")
    print(f"    p-value: {shapiro_p:.4e}")
    print(f"    Normal: {'Yes' if shapiro_p > 0.05 else 'No'} (at α=0.05)")

    # D'Agostino-Pearson test
    dagostino_stat, dagostino_p = normaltest(delta_pce)
    print(f"\n  D'Agostino-Pearson Test:")
    print(f"    Statistic: {dagostino_stat:.4f}")
    print(f"    p-value: {dagostino_p:.4e}")
    print(f"    Normal: {'Yes' if dagostino_p > 0.05 else 'No'} (at α=0.05)")

    # Create figure with histogram and KDE
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Histogram with KDE
    ax1 = axes[0, 0]
    sns.histplot(delta_pce, kde=True, bins=50, color='steelblue', ax=ax1, stat='density')
    ax1.set_xlabel('Delta_PCE (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution of Delta_PCE with KDE', fontsize=14, fontweight='bold')

    # Add vertical lines for mean and median
    ax1.axvline(stats_summary['Mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_summary["Mean"]:.3f}')
    ax1.axvline(stats_summary['Median'], color='green', linestyle='-.', linewidth=2, label=f'Median: {stats_summary["Median"]:.3f}')
    ax1.legend()

    # Q-Q plot
    ax2 = axes[0, 1]
    stats.probplot(delta_pce, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot for Delta_PCE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax2.set_ylabel('Sample Quantiles', fontsize=12)

    # Box plot
    ax3 = axes[1, 0]
    sns.boxplot(x=delta_pce, color='lightcoral', ax=ax3)
    ax3.set_xlabel('Delta_PCE (%)', fontsize=12)
    ax3.set_title('Box Plot of Delta_PCE', fontsize=14, fontweight='bold')

    # Violin plot
    ax4 = axes[1, 1]
    sns.violinplot(x=delta_pce, color='lightseagreen', ax=ax4)
    ax4.set_xlabel('Delta_PCE (%)', fontsize=12)
    ax4.set_title('Violin Plot of Delta_PCE', fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'delta_pce_distribution.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    return stats_summary, {'shapiro': (shapiro_stat, shapiro_p), 'dagostino': (dagostino_stat, dagostino_p)}

def analyze_descriptor_distributions(df):
    """Analyze distributions of all chemical descriptors."""
    print("\n" + "=" * 60)
    print("STEP 3: Analyzing Chemical Descriptor Distributions")
    print("=" * 60)

    descriptor_stats = {}

    for col in CHEMICAL_DESCRIPTORS:
        data = df[col].dropna()
        descriptor_stats[col] = {
            'Count': len(data),
            'Mean': data.mean(),
            'Median': data.median(),
            'Std': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Skewness': skew(data),
            'Kurtosis': kurtosis(data)
        }

        # Determine if highly skewed
        abs_skew = abs(descriptor_stats[col]['Skewness'])
        if abs_skew > 1:
            skew_level = 'Highly skewed'
        elif abs_skew > 0.5:
            skew_level = 'Moderately skewed'
        else:
            skew_level = 'Approximately symmetric'

        descriptor_stats[col]['Skewness_Level'] = skew_level

        print(f"\n{col}:")
        print(f"  Mean: {descriptor_stats[col]['Mean']:.4f}")
        print(f"  Std: {descriptor_stats[col]['Std']:.4f}")
        print(f"  Skewness: {descriptor_stats[col]['Skewness']:.4f} ({skew_level})")
        print(f"  Kurtosis: {descriptor_stats[col]['Kurtosis']:.4f}")

    # Create multi-panel figure with histograms
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, col in enumerate(CHEMICAL_DESCRIPTORS):
        ax = axes[idx]
        data = df[col].dropna()

        # Histogram with KDE
        sns.histplot(data, kde=True, bins=30, color='steelblue', ax=ax, stat='density')

        # Format column name for title
        title = col.replace('_', ' ').title()
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Distribution of {title}\n(Skewness: {descriptor_stats[col]["Skewness"]:.2f})',
                     fontsize=12, fontweight='bold')

    plt.suptitle('Distribution of Chemical Descriptors', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, 'descriptor_distributions.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    return descriptor_stats

def create_boxplots_by_quartile(df):
    """Create box plots for chemical descriptors grouped by Delta_PCE quartiles."""
    print("\n" + "=" * 60)
    print("STEP 4: Creating Box Plots by Delta_PCE Quartiles")
    print("=" * 60)

    # Create quartile labels
    df_temp = df.copy()
    df_temp['Delta_PCE_Quartile'] = pd.qcut(df_temp['Delta_PCE'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

    print("\nDelta_PCE Quartile Ranges:")
    quartile_ranges = df_temp.groupby('Delta_PCE_Quartile')['Delta_PCE'].agg(['min', 'max', 'count'])
    print(quartile_ranges)

    # Create box plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, col in enumerate(CHEMICAL_DESCRIPTORS):
        ax = axes[idx]

        # Box plot
        sns.boxplot(data=df_temp, x='Delta_PCE_Quartile', y=col, palette='Set2', ax=ax)

        # Format column name
        ylabel = col.replace('_', ' ').title()
        ax.set_xlabel('Delta_PCE Quartile', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel} by Delta_PCE Quartile', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Chemical Descriptors Grouped by Delta_PCE Quartiles', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, 'boxplots_by_quartile.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    return quartile_ranges

def detect_outliers(df):
    """Perform outlier detection using IQR and Z-score methods."""
    print("\n" + "=" * 60)
    print("STEP 5: Outlier Detection")
    print("=" * 60)

    all_columns = CHEMICAL_DESCRIPTORS + ['Delta_PCE']
    outlier_summary = {}

    for col in all_columns:
        data = df[col].dropna()

        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        iqr_percentage = (iqr_outliers / len(data)) * 100

        # Z-score method (threshold = 3)
        z_scores = np.abs(stats.zscore(data))
        zscore_outliers = (z_scores > 3).sum()
        zscore_percentage = (zscore_outliers / len(data)) * 100

        outlier_summary[col] = {
            'IQR_Lower': lower_bound,
            'IQR_Upper': upper_bound,
            'IQR_Outliers': iqr_outliers,
            'IQR_Percentage': iqr_percentage,
            'Zscore_Outliers': zscore_outliers,
            'Zscore_Percentage': zscore_percentage
        }

        print(f"\n{col}:")
        print(f"  IQR Method:")
        print(f"    Lower bound: {lower_bound:.4f}")
        print(f"    Upper bound: {upper_bound:.4f}")
        print(f"    Outliers: {iqr_outliers} ({iqr_percentage:.2f}%)")
        print(f"  Z-score Method (|z| > 3):")
        print(f"    Outliers: {zscore_outliers} ({zscore_percentage:.2f}%)")

    return outlier_summary

def save_analysis_report(delta_pce_stats, normality_tests, descriptor_stats,
                         quartile_ranges, outlier_summary):
    """Save comprehensive analysis report."""
    print("\n" + "=" * 60)
    print("STEP 6: Saving Analysis Report")
    print("=" * 60)

    report_path = os.path.join(OUTPUT_DIR, 'distribution_analysis.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DISTRIBUTION ANALYSIS REPORT\n")
        f.write("QSPR Data Analysis for Perovskite Solar Cells\n")
        f.write("Date: 2026-02-20\n")
        f.write("=" * 80 + "\n\n")

        # Delta_PCE Analysis
        f.write("-" * 80 + "\n")
        f.write("1. DELTA_PCE DISTRIBUTION ANALYSIS\n")
        f.write("-" * 80 + "\n\n")

        f.write("Basic Statistics:\n")
        for key, value in delta_pce_stats.items():
            f.write(f"  {key}: {value:.4f}\n")

        f.write("\nNormality Tests:\n")
        f.write(f"  Shapiro-Wilk Test:\n")
        f.write(f"    Statistic: {normality_tests['shapiro'][0]:.4f}\n")
        f.write(f"    p-value: {normality_tests['shapiro'][1]:.4e}\n")
        f.write(f"    Result: {'Normal' if normality_tests['shapiro'][1] > 0.05 else 'Non-normal'} (at α=0.05)\n")
        f.write(f"\n  D'Agostino-Pearson Test:\n")
        f.write(f"    Statistic: {normality_tests['dagostino'][0]:.4f}\n")
        f.write(f"    p-value: {normality_tests['dagostino'][1]:.4e}\n")
        f.write(f"    Result: {'Normal' if normality_tests['dagostino'][1] > 0.05 else 'Non-normal'} (at α=0.05)\n")

        # Chemical Descriptors Analysis
        f.write("\n" + "-" * 80 + "\n")
        f.write("2. CHEMICAL DESCRIPTOR DISTRIBUTIONS\n")
        f.write("-" * 80 + "\n\n")

        for col, stats_dict in descriptor_stats.items():
            f.write(f"{col}:\n")
            f.write(f"  Mean: {stats_dict['Mean']:.4f}\n")
            f.write(f"  Median: {stats_dict['Median']:.4f}\n")
            f.write(f"  Std: {stats_dict['Std']:.4f}\n")
            f.write(f"  Min: {stats_dict['Min']:.4f}\n")
            f.write(f"  Max: {stats_dict['Max']:.4f}\n")
            f.write(f"  Skewness: {stats_dict['Skewness']:.4f} ({stats_dict['Skewness_Level']})\n")
            f.write(f"  Kurtosis: {stats_dict['Kurtosis']:.4f}\n\n")

        # Quartile Analysis
        f.write("-" * 80 + "\n")
        f.write("3. DELTA_PCE QUARTILE ANALYSIS\n")
        f.write("-" * 80 + "\n\n")

        f.write(quartile_ranges.to_string())
        f.write("\n\n")

        # Outlier Analysis
        f.write("-" * 80 + "\n")
        f.write("4. OUTLIER DETECTION SUMMARY\n")
        f.write("-" * 80 + "\n\n")

        for col, outliers in outlier_summary.items():
            f.write(f"{col}:\n")
            f.write(f"  IQR Method: {outliers['IQR_Outliers']} outliers ({outliers['IQR_Percentage']:.2f}%)\n")
            f.write(f"    Bounds: [{outliers['IQR_Lower']:.4f}, {outliers['IQR_Upper']:.4f}]\n")
            f.write(f"  Z-score Method: {outliers['Zscore_Outliers']} outliers ({outliers['Zscore_Percentage']:.2f}%)\n\n")

        # Key Findings
        f.write("-" * 80 + "\n")
        f.write("5. KEY FINDINGS\n")
        f.write("-" * 80 + "\n\n")

        # Identify highly skewed features
        highly_skewed = [col for col, stats in descriptor_stats.items()
                         if abs(stats['Skewness']) > 1]
        if highly_skewed:
            f.write("Highly skewed chemical descriptors (|skewness| > 1):\n")
            for col in highly_skewed:
                f.write(f"  - {col} (skewness: {descriptor_stats[col]['Skewness']:.4f})\n")
        else:
            f.write("No highly skewed chemical descriptors found.\n")

        f.write("\n")

        # Normality conclusion
        if normality_tests['shapiro'][1] < 0.05 or normality_tests['dagostino'][1] < 0.05:
            f.write("Delta_PCE distribution is non-normal.\n")
            f.write("Recommendation: Consider using non-parametric statistical methods or data transformation.\n")
        else:
            f.write("Delta_PCE distribution is approximately normal.\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Report saved: {report_path}")
    return report_path

def main():
    """Main function to run the distribution analysis."""
    print("\n" + "=" * 60)
    print("QSPR DISTRIBUTION ANALYSIS FOR PEROVSKITE SOLAR CELLS")
    print("=" * 60)

    # Create figures directory if it doesn't exist
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load and preprocess data
    df = load_and_preprocess_data()

    # Analyze Delta_PCE distribution
    delta_pce_stats, normality_tests = analyze_delta_pce_distribution(df)

    # Analyze chemical descriptor distributions
    descriptor_stats = analyze_descriptor_distributions(df)

    # Create box plots by quartile
    quartile_ranges = create_boxplots_by_quartile(df)

    # Detect outliers
    outlier_summary = detect_outliers(df)

    # Save analysis report
    report_path = save_analysis_report(
        delta_pce_stats, normality_tests, descriptor_stats,
        quartile_ranges, outlier_summary
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - Processed data: {PROCESSED_DATA_PATH}")
    print(f"  - Figures directory: {FIGURES_DIR}/")
    print(f"  - Analysis report: {report_path}")
    print("\nFigure files:")
    for fig in ['delta_pce_distribution.png', 'descriptor_distributions.png', 'boxplots_by_quartile.png']:
        print(f"  - {os.path.join(FIGURES_DIR, fig)}")

if __name__ == "__main__":
    main()
