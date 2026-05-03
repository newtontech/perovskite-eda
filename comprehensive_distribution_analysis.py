#!/usr/bin/env python3
"""
Comprehensive Distribution Analysis for QSPR Data
Analyzes Delta_PCE and chemical descriptor distributions with advanced statistical tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, skew, kurtosis, moment
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/share/yhm/test/AutoML_EDA/processed_data.csv'
OUTPUT_DIR = '/share/yhm/test/AutoML_EDA'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')

# Chemical descriptor columns
DESCRIPTOR_COLS = ['molecular_weight', 'h_bond_donors', 'h_bond_acceptors',
                   'rotatable_bonds', 'tpsa', 'log_p']

# Create output directories
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

print("=" * 80)
print("COMPREHENSIVE DISTRIBUTION ANALYSIS FOR QSPR DATA")
print("=" * 80)

# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================
print("\n1. Loading processed data...")
df = pd.read_csv(DATA_PATH)

# Remove rows with missing Delta_PCE
df_clean = df.dropna(subset=['Delta_PCE']).copy()
print(f"   Total samples: {len(df)}")
print(f"   Samples with Delta_PCE: {len(df_clean)}")
print(f"   Columns: {list(df.columns)}")

# ============================================================================
# SECTION 2: DELTA_PCE DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. DELTA_PCE DISTRIBUTION ANALYSIS")
print("=" * 80)

delta_pce = df_clean['Delta_PCE'].dropna()
print(f"   Valid Delta_PCE values: {len(delta_pce)}")
print(f"   Range: [{delta_pce.min():.3f}, {delta_pce.max():.3f}]")
print(f"   Mean: {delta_pce.mean():.3f}, Median: {delta_pce.median():.3f}")
print(f"   Std: {delta_pce.std():.3f}")

# Figure 1: Histogram with KDE and normal distribution overlay
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Histogram with KDE
ax1 = axes[0]
ax1.hist(delta_pce, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.5)
from scipy.stats import gaussian_kde
kde = gaussian_kde(delta_pce)
x_range = np.linspace(delta_pce.min(), delta_pce.max(), 500)
ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

# Overlay normal distribution
mu, sigma = delta_pce.mean(), delta_pce.std()
normal_dist = stats.norm(loc=mu, scale=sigma)
ax1.plot(x_range, normal_dist.pdf(x_range), 'g--', linewidth=2, label='Normal Fit')
ax1.axvline(delta_pce.mean(), color='red', linestyle=':', linewidth=2, label=f'Mean: {mu:.2f}')
ax1.axvline(delta_pce.median(), color='orange', linestyle=':', linewidth=2, label=f'Median: {delta_pce.median():.2f}')
ax1.set_xlabel('Delta_PCE (%)')
ax1.set_ylabel('Density')
ax1.set_title('Delta_PCE Distribution with KDE and Normal Fit')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Q-Q plot
ax2 = axes[1]
stats.probplot(delta_pce, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot: Delta_PCE vs Normal Distribution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'distribution_delta_pce_histogram_qq.png'), bbox_inches='tight')
print("   Saved: figures/distribution_delta_pce_histogram_qq.png")
plt.close()

# Statistical tests for normality
print("\n   Normality Tests:")
print("   " + "-" * 60)

# Shapiro-Wilk test (for n < 5000, use sample)
if len(delta_pce) <= 5000:
    shapiro_stat, shapiro_p = shapiro(delta_pce)
else:
    sample_delta = np.random.choice(delta_pce, 5000, replace=False)
    shapiro_stat, shapiro_p = shapiro(sample_delta)
print(f"   Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4e}")
print(f"   → {'Normal' if shapiro_p > 0.05 else 'NOT Normal'} (α=0.05)")

# D'Agostino-Pearson test
dagostino_stat, dagostino_p = normaltest(delta_pce)
print(f"   D'Agostino-Pearson: χ²={dagostino_stat:.4f}, p={dagostino_p:.4e}")
print(f"   → {'Normal' if dagostino_p > 0.05 else 'NOT Normal'} (α=0.05)")

# Distribution moments
print("\n   Distribution Moments:")
print("   " + "-" * 60)
print(f"   Skewness: {skew(delta_pce):.4f}")
print(f"   Kurtosis: {kurtosis(delta_pce):.4f} (excess kurtosis)")
print(f"   1st Moment (Mean): {moment(delta_pce, moment=1):.4f}")
print(f"   2nd Moment (Variance): {moment(delta_pce, moment=2):.4f}")
print(f"   3rd Moment: {moment(delta_pce, moment=3):.4f}")
print(f"   4th Moment: {moment(delta_pce, moment=4):.4f}")

# ============================================================================
# SECTION 3: CHEMICAL DESCRIPTOR DISTRIBUTIONS
# ============================================================================
print("\n" + "=" * 80)
print("3. CHEMICAL DESCRIPTOR DISTRIBUTIONS")
print("=" * 80)

# Prepare descriptor data
desc_data = df_clean[DESCRIPTOR_COLS].dropna()
print(f"   Samples with complete descriptors: {len(desc_data)}")

# Summary statistics
desc_stats = desc_data.describe()
desc_stats.loc['skewness'] = desc_data.apply(skew)
desc_stats.loc['kurtosis'] = desc_data.apply(kurtosis)
desc_stats.to_csv(os.path.join(TABLES_DIR, 'distribution_statistics.csv'))
print("   Saved: tables/distribution_statistics.csv")

# Figure 2: Multi-panel histogram grid with KDE
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(DESCRIPTOR_COLS):
    ax = axes[i]
    data = desc_data[col].dropna()

    # Histogram with KDE
    ax.hist(data, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.3)

    # KDE
    if len(data) > 1:
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 300)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # Statistics annotation
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {data.mean():.2f}')
    ax.axvline(data.median(), color='green', linestyle='--', linewidth=1.5, label=f'Median: {data.median():.2f}')

    ax.set_xlabel(col.replace('_', ' ').title())
    ax.set_ylabel('Density')
    ax.set_title(f'{col.replace("_", " ").title()}\n(Skew: {skew(data):.2f}, Kurt: {kurtosis(data):.2f})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'distribution_descriptors_histograms.png'), bbox_inches='tight')
print("   Saved: figures/distribution_descriptors_histograms.png")
plt.close()

# Figure 3: Box plots and violin plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plot
ax1 = axes[0]
box_data = [desc_data[col].dropna().values for col in DESCRIPTOR_COLS]
bp = ax1.boxplot(box_data, labels=[col.replace('_', '\n') for col in DESCRIPTOR_COLS],
                 patch_artist=True, notch=True, showmeans=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax1.set_ylabel('Value')
ax1.set_title('Box Plots of Chemical Descriptors')
ax1.grid(True, alpha=0.3, axis='y')

# Violin plot
ax2 = axes[1]
positions = np.arange(1, len(DESCRIPTOR_COLS) + 1)
vp = ax2.violinplot(box_data, positions=positions, showmeans=True, showmedians=True, showextrema=True)
ax2.set_xticks(positions)
ax2.set_xticklabels([col.replace('_', '\n') for col in DESCRIPTOR_COLS])
ax2.set_ylabel('Value')
ax2.set_title('Violin Plots of Chemical Descriptors')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'distribution_descriptors_box_violin.png'), bbox_inches='tight')
print("   Saved: figures/distribution_descriptors_box_violin.png")
plt.close()

# Log-transform analysis for skewed features
print("\n   Log-Transform Analysis for Skewed Features:")
print("   " + "-" * 60)

skewed_features = []
for col in DESCRIPTOR_COLS:
    data = desc_data[col].dropna()
    sk = skew(data)
    if abs(sk) > 1.0:  # Highly skewed
        skewed_features.append(col)
        print(f"   {col}: skewness = {sk:.3f} (HIGHLY skewed)")

if skewed_features:
    # Create log-transform analysis figure
    fig, axes = plt.subplots(len(skewed_features), 2, figsize=(12, 4*len(skewed_features)))
    if len(skewed_features) == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(skewed_features):
        data = desc_data[col].dropna()

        # Original
        ax1 = axes[i, 0]
        ax1.hist(data, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='black')
        ax1.set_title(f'{col.replace("_", " ").title()} (Original)\nSkew: {skew(data):.3f}')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)

        # Log-transformed (add small constant if needed)
        if data.min() > 0:
            log_data = np.log(data)
            title_suffix = f'Log-transformed\nSkew: {skew(log_data):.3f}'
        else:
            log_data = np.log(data - data.min() + 1)
            title_suffix = f'Log(x-min+1)\nSkew: {skew(log_data):.3f}'

        ax2 = axes[i, 1]
        ax2.hist(log_data, bins=40, density=True, alpha=0.6, color='coral', edgecolor='black')
        ax2.set_title(title_suffix)
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'distribution_log_transform_analysis.png'), bbox_inches='tight')
    print("   Saved: figures/distribution_log_transform_analysis.png")
    plt.close()
else:
    print("   No highly skewed features found (|skewness| > 1.0)")

# ============================================================================
# SECTION 4: OUTLIER DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("4. OUTLIER DETECTION")
print("=" * 80)

# Prepare data for outlier analysis (complete cases only)
outlier_data = df_clean[DESCRIPTOR_COLS + ['Delta_PCE']].dropna()
print(f"   Complete cases: {len(outlier_data)}")

outlier_results = {}

# 4.1: IQR Method
print("\n   4.1 IQR Method (1.5 × IQR):")
print("   " + "-" * 60)
iqr_outliers = {}
for col in outlier_data.columns:
    Q1 = outlier_data[col].quantile(0.25)
    Q3 = outlier_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = (outlier_data[col] < lower) | (outlier_data[col] > upper)
    iqr_outliers[col] = outliers
    n_outliers = outliers.sum()
    print(f"   {col}: {n_outliers} outliers ({100*n_outliers/len(outlier_data):.2f}%)")

outlier_results['IQR'] = iqr_outliers

# 4.2: Z-score Method
print("\n   4.2 Z-Score Method (|z| > 3):")
print("   " + "-" * 60)
zscore_outliers = {}
for col in outlier_data.columns:
    z_scores = np.abs((outlier_data[col] - outlier_data[col].mean()) / outlier_data[col].std())
    outliers = z_scores > 3
    zscore_outliers[col] = outliers
    n_outliers = outliers.sum()
    print(f"   {col}: {n_outliers} outliers ({100*n_outliers/len(outlier_data):.2f}%)")

outlier_results['Zscore'] = zscore_outliers

# 4.3: Isolation Forest (multivariate)
print("\n   4.3 Isolation Forest (Multivariate):")
print("   " + "-" * 60)

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(outlier_data)

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_pred = iso_forest.fit_predict(scaled_data)
outlier_scores = iso_forest.score_samples(scaled_data)

n_outliers = (outlier_pred == -1).sum()
print(f"   Detected: {n_outliers} outliers ({100*n_outliers/len(outlier_data):.2f}%)")

outlier_results['IsolationForest'] = pd.Series(outlier_pred == -1, index=outlier_data.index)

# 4.4: PCA Visualization
print("\n   4.4 PCA Visualization:")

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

explained_var = pca.explained_variance_ratio_ * 100
print(f"   PC1: {explained_var[0]:.2f}% variance")
print(f"   PC2: {explained_var[1]:.2f}% variance")

# Figure 4: Outlier detection visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 4a: PCA colored by Delta_PCE
ax1 = axes[0, 0]
scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1],
                      c=outlier_data['Delta_PCE'], cmap='RdYlGn_r',
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax1.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)')
ax1.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)')
ax1.set_title('PCA Colored by Delta_PCE')
plt.colorbar(scatter, ax=ax1, label='Delta_PCE')
ax1.grid(True, alpha=0.3)

# 4b: PCA with Isolation Forest outliers highlighted
ax2 = axes[0, 1]
colors = ['blue' if x == 1 else 'red' for x in outlier_pred]
ax2.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax2.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)')
ax2.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)')
ax2.set_title('PCA with Isolation Forest Outliers (Red)')
ax2.grid(True, alpha=0.3)

# 4c: Outlier scores
ax3 = axes[1, 0]
scatter3 = ax3.scatter(pca_result[:, 0], pca_result[:, 1],
                       c=-outlier_scores, cmap='Reds',
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax3.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)')
ax3.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)')
ax3.set_title('PCA Colored by Outlier Score')
plt.colorbar(scatter3, ax=ax3, label='Outlier Score')
ax3.grid(True, alpha=0.3)

# 4d: Combined outlier summary
ax4 = axes[1, 1]
# Count outliers per method
methods = []
counts = []
for col in outlier_data.columns:
    methods.append(f'{col}\n(IQR)')
    counts.append(iqr_outliers[col].sum())
    methods.append(f'{col}\n(Z-score)')
    counts.append(zscore_outliers[col].sum())

y_pos = np.arange(len(methods))
bars = ax4.barh(y_pos, counts, color='steelblue', alpha=0.7)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(methods, fontsize=7)
ax4.set_xlabel('Number of Outliers')
ax4.set_title('Outlier Counts by Method and Feature')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'outlier_detection_summary.png'), bbox_inches='tight')
print("   Saved: figures/outlier_detection_summary.png")
plt.close()

# Save outlier indices
outlier_summary = pd.DataFrame({
    'IQR_outliers': [iqr_outliers[c].sum() for c in outlier_data.columns],
    'Zscore_outliers': [zscore_outliers[c].sum() for c in outlier_data.columns]
}, index=outlier_data.columns)
outlier_summary.to_csv(os.path.join(TABLES_DIR, 'outlier_summary.csv'))
print("   Saved: tables/outlier_summary.csv")

# ============================================================================
# SECTION 5: DELTA_PCE QUARTILE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. DELTA_PCE QUARTILE ANALYSIS")
print("=" * 80)

# Create quartiles
quartile_data = df_clean.dropna(subset=DESCRIPTOR_COLS + ['Delta_PCE']).copy()
quartile_data['Delta_PCE_quartile'] = pd.qcut(quartile_data['Delta_PCE'],
                                               q=4,
                                               labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

print(f"   Samples per quartile:")
print(quartile_data['Delta_PCE_quartile'].value_counts().sort_index())

# Quartile statistics
quartile_stats = quartile_data.groupby('Delta_PCE_quartile')[DESCRIPTOR_COLS].agg(['mean', 'std', 'median'])
print("\n   Descriptor Statistics by Quartile:")
print(quartile_stats)

# ANOVA tests
print("\n   ANOVA Tests (p-values):")
print("   " + "-" * 60)
anova_results = {}

for col in DESCRIPTOR_COLS:
    # Get groups
    groups = [quartile_data[quartile_data['Delta_PCE_quartile'] == q][col].values
              for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']]

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    anova_results[col] = {'F_statistic': f_stat, 'p_value': p_value}

    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"   {col}: F={f_stat:.4f}, p={p_value:.4e} {significance}")

# Figure 5: Quartile comparison plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(DESCRIPTOR_COLS):
    ax = axes[i]

    # Box plot by quartile
    quartile_data.boxplot(column=col, by='Delta_PCE_quartile', ax=ax)
    ax.set_xlabel('Delta_PCE Quartile')
    ax.set_ylabel(col.replace('_', ' ').title())
    ax.set_title(f'{col.replace("_", " ").title()}\nANOVA p={anova_results[col]["p_value"]:.2e}')
    ax.grid(True, alpha=0.3)

plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'distribution_quartile_comparison.png'), bbox_inches='tight')
print("   Saved: figures/distribution_quartile_comparison.png")
plt.close()

# Figure 6: Quartile distribution overlay for Delta_PCE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Kernel density by quartile for Delta_PCE
ax1 = axes[0]
for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
    data = quartile_data[quartile_data['Delta_PCE_quartile'] == quartile]['Delta_PCE']
    if len(data) > 1:
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax1.plot(x_range, kde(x_range), label=quartile, linewidth=2)

ax1.set_xlabel('Delta_PCE (%)')
ax1.set_ylabel('Density')
ax1.set_title('Delta_PCE Distribution by Quartile')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Mean Delta_PCE by quartile with error bars
ax2 = axes[1]
quartile_means = quartile_data.groupby('Delta_PCE_quartile')['Delta_PCE'].agg(['mean', 'std', 'count'])
quartile_means['sem'] = quartile_means['std'] / np.sqrt(quartile_means['count'])

x_pos = np.arange(4)
bars = ax2.bar(x_pos, quartile_means['mean'], yerr=quartile_means['sem'],
               capsize=5, color=['#d73027', '#fee08b', '#fee08b', '#1a9850'], alpha=0.7, edgecolor='black')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])
ax2.set_ylabel('Mean Delta_PCE (%)')
ax2.set_title('Mean Delta_PCE by Quartile (±SEM)')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'distribution_quartile_delta_pce.png'), bbox_inches='tight')
print("   Saved: figures/distribution_quartile_delta_pce.png")
plt.close()

# Save quartile statistics
quartile_stats.to_csv(os.path.join(TABLES_DIR, 'quartile_statistics.csv'))
print("   Saved: tables/quartile_statistics.csv")

# Save ANOVA results
anova_df = pd.DataFrame(anova_results).T
anova_df.to_csv(os.path.join(TABLES_DIR, 'anova_results.csv'))
print("   Saved: tables/anova_results.csv")

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 80)
print("6. GENERATING FINAL REPORT")
print("=" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("COMPREHENSIVE DISTRIBUTION ANALYSIS REPORT FOR QSPR DATA")
report_lines.append("=" * 80)
report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")

# Data Summary
report_lines.append("1. DATA SUMMARY")
report_lines.append("-" * 80)
report_lines.append(f"Total samples: {len(df)}")
report_lines.append(f"Samples with Delta_PCE: {len(df_clean)}")
report_lines.append(f"Complete cases (all descriptors): {len(outlier_data)}")
report_lines.append("")

# Delta_PCE Distribution
report_lines.append("2. DELTA_PCE DISTRIBUTION")
report_lines.append("-" * 80)
report_lines.append(f"Mean: {delta_pce.mean():.4f}")
report_lines.append(f"Median: {delta_pce.median():.4f}")
report_lines.append(f"Std Dev: {delta_pce.std():.4f}")
report_lines.append(f"Range: [{delta_pce.min():.4f}, {delta_pce.max():.4f}]")
report_lines.append(f"Skewness: {skew(delta_pce):.4f}")
report_lines.append(f"Kurtosis: {kurtosis(delta_pce):.4f}")
report_lines.append("")
report_lines.append("Normality Tests:")
report_lines.append(f"  Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4e}")
report_lines.append(f"  D'Agostino-Pearson: χ²={dagostino_stat:.4f}, p={dagostino_p:.4e}")
report_lines.append("")

# Descriptor Statistics
report_lines.append("3. CHEMICAL DESCRIPTOR STATISTICS")
report_lines.append("-" * 80)
for col in DESCRIPTOR_COLS:
    data = desc_data[col].dropna()
    report_lines.append(f"{col}:")
    report_lines.append(f"  Mean: {data.mean():.4f}, Median: {data.median():.4f}, Std: {data.std():.4f}")
    report_lines.append(f"  Skewness: {skew(data):.4f}, Kurtosis: {kurtosis(data):.4f}")
    report_lines.append(f"  Range: [{data.min():.4f}, {data.max():.4f}]")
report_lines.append("")

# Outlier Summary
report_lines.append("4. OUTLIER DETECTION SUMMARY")
report_lines.append("-" * 80)
report_lines.append("IQR Method (1.5 × IQR):")
for col in outlier_data.columns:
    report_lines.append(f"  {col}: {iqr_outliers[col].sum()} outliers")
report_lines.append("")
report_lines.append("Z-Score Method (|z| > 3):")
for col in outlier_data.columns:
    report_lines.append(f"  {col}: {zscore_outliers[col].sum()} outliers")
report_lines.append("")
report_lines.append(f"Isolation Forest: {n_outliers} outliers ({100*n_outliers/len(outlier_data):.1f}%)")
report_lines.append("")

# ANOVA Results
report_lines.append("5. QUARTILE ANALYSIS - ANOVA RESULTS")
report_lines.append("-" * 80)
for col, results in anova_results.items():
    sig = "***" if results['p_value'] < 0.001 else "**" if results['p_value'] < 0.01 else "*" if results['p_value'] < 0.05 else "ns"
    report_lines.append(f"{col}: F={results['F_statistic']:.4f}, p={results['p_value']:.4e} {sig}")
report_lines.append("")

# Files Created
report_lines.append("6. OUTPUT FILES")
report_lines.append("-" * 80)
report_lines.append("Figures:")
report_lines.append("  figures/distribution_delta_pce_histogram_qq.png")
report_lines.append("  figures/distribution_descriptors_histograms.png")
report_lines.append("  figures/distribution_descriptors_box_violin.png")
report_lines.append("  figures/distribution_log_transform_analysis.png")
report_lines.append("  figures/outlier_detection_summary.png")
report_lines.append("  figures/distribution_quartile_comparison.png")
report_lines.append("  figures/distribution_quartile_delta_pce.png")
report_lines.append("")
report_lines.append("Tables:")
report_lines.append("  tables/distribution_statistics.csv")
report_lines.append("  tables/outlier_summary.csv")
report_lines.append("  tables/quartile_statistics.csv")
report_lines.append("  tables/anova_results.csv")
report_lines.append("")
report_lines.append("=" * 80)

# Write report
with open(os.path.join(OUTPUT_DIR, 'distribution_analysis_report.txt'), 'w') as f:
    f.write('\n'.join(report_lines))

print("   Report saved: distribution_analysis_report.txt")

# Print summary
print("\n" + "=" * 80)
print("DISTRIBUTION ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nTotal figures created: 7")
print(f"Total tables created: 4")
print(f"Report: distribution_analysis_report.txt")
print("\nAll results saved to /share/yhm/test/AutoML_EDA/")
