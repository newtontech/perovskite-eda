#!/usr/bin/env python3
"""
Comprehensive QSPR Correlation Analysis for Perovskite Solar Cell Modulators

This script performs statistical analysis and visualization of relationships between
chemical descriptors and Delta_PCE in perovskite solar cell modulators.

Author: AutoML-EDA Analysis
Date: 2026-02-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import warnings
import textwrap
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Set color palette for scientific publications
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
sns.set_palette(colors)

def load_and_prepare_data(filepath):
    """Load and prepare the dataset for analysis."""
    print("=" * 80)
    print("STEP 1: Loading and Preparing Data")
    print("=" * 80)

    df = pd.read_csv(filepath)
    print(f"Original dataset shape: {df.shape}")

    # Define chemical features and target
    chemical_features = [
        'molecular_weight',
        'h_bond_donors',
        'h_bond_acceptors',
        'rotatable_bonds',
        'tpsa',
        'log_p'
    ]

    target = 'Delta_PCE'

    # Check for missing values
    missing_data = df[chemical_features + [target]].isnull().sum()
    print("\nMissing values per column:")
    for col, count in missing_data.items():
        print(f"  {col}: {count}")

    # Filter rows with missing target values
    df_clean = df.dropna(subset=[target] + chemical_features).copy()
    print(f"\nDataset shape after removing NaN values: {df_clean.shape}")

    # Display summary statistics
    print("\nSummary Statistics for Chemical Descriptors:")
    print("-" * 80)
    summary_stats = df_clean[chemical_features + [target]].describe()
    print(summary_stats.to_string())

    return df_clean, chemical_features, target

def calculate_correlations(df, features, target):
    """Calculate Pearson, Spearman, and Kendall correlations with significance tests."""
    print("\n" + "=" * 80)
    print("STEP 2: Calculating Correlations and Statistical Significance")
    print("=" * 80)

    results = []

    for feature in features:
        # Remove NaN pairs for this specific comparison
        valid_idx = df[[feature, target]].notna().all(axis=1)
        x = df.loc[valid_idx, feature]
        y = df.loc[valid_idx, target]
        n = len(x)

        # Calculate correlations
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        kendall_tau, kendall_p = kendalltau(x, y)

        # Calculate 95% confidence intervals for Pearson r
        # Fisher's z-transformation
        z_r = np.arctanh(pearson_r)
        se = 1 / np.sqrt(n - 3)
        z_lower = z_r - 1.96 * se
        z_upper = z_r + 1.96 * se
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)

        results.append({
            'Feature': feature,
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Pearson_CI_lower': r_lower,
            'Pearson_CI_upper': r_upper,
            'Spearman_rho': spearman_r,
            'Spearman_p': spearman_p,
            'Kendall_tau': kendall_tau,
            'Kendall_p': kendall_p,
            'N': n
        })

        print(f"\n{feature}:")
        print(f"  Pearson r: {pearson_r:.4f} (p-value: {pearson_p:.4e})")
        print(f"  Spearman ρ: {spearman_r:.4f} (p-value: {spearman_p:.4e})")
        print(f"  Kendall τ: {kendall_tau:.4f} (p-value: {kendall_p:.4e})")
        print(f"  95% CI for Pearson: [{r_lower:.4f}, {r_upper:.4f}]")

    results_df = pd.DataFrame(results)

    # Apply Bonferroni correction for multiple comparisons
    print("\n" + "-" * 80)
    print("Multiple Testing Correction (Bonferroni)")
    print("-" * 80)

    n_tests = len(features)
    results_df['Pearson_p_bonf'] = results_df['Pearson_p'] * n_tests
    results_df['Spearman_p_bonf'] = results_df['Spearman_p'] * n_tests
    results_df['Kendall_p_bonf'] = results_df['Kendall_p'] * n_tests

    # Cap at 1.0
    for col in ['Pearson_p_bonf', 'Spearman_p_bonf', 'Kendall_p_bonf']:
        results_df[col] = results_df[col].clip(upper=1.0)

    print(f"\nNumber of tests: {n_tests}")
    print(f"Bonferroni correction factor: {n_tests}")
    print("\nSignificance after Bonferroni correction:")
    print("Pearson:")
    for _, row in results_df.iterrows():
        sig = "***" if row['Pearson_p_bonf'] < 0.001 else "**" if row['Pearson_p_bonf'] < 0.01 else "*" if row['Pearson_p_bonf'] < 0.05 else "ns"
        print(f"  {row['Feature']}: p_adj = {row['Pearson_p_bonf']:.4e} {sig}")

    return results_df

def create_correlation_heatmap(df, features, target, output_path):
    """Create a correlation matrix heatmap."""
    print("\n" + "=" * 80)
    print("STEP 3: Creating Correlation Matrix Heatmap")
    print("=" * 80)

    # Calculate correlation matrix
    corr_matrix = df[features + [target]].corr(method='pearson')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
                vmin=-1, vmax=1,
                ax=ax)

    ax.set_title('Correlation Matrix: Chemical Descriptors and Delta_PCE\n',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Correlation heatmap saved to: {output_path}")

    # Save correlation matrix
    corr_path = output_path.replace('figures', 'tables').replace('.png', '_matrix.csv')
    corr_matrix.to_csv(corr_path)
    print(f"Correlation matrix saved to: {corr_path}")

    return corr_matrix

def create_scatter_plots_with_regression(df, features, target, output_path):
    """Create scatter plots with regression lines for each feature."""
    print("\n" + "=" * 80)
    print("STEP 4: Creating Scatter Plots with Regression Lines")
    print("=" * 80)

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features):
        ax = axes[idx]

        # Remove NaN pairs
        valid_idx = df[[feature, target]].notna().all(axis=1)
        x = df.loc[valid_idx, feature]
        y = df.loc[valid_idx, target]

        # Calculate correlation
        r, p = pearsonr(x, y)

        # Create scatter plot
        scatter = ax.scatter(x, y, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)

        # Add regression line
        z = np.polyfit(x, y, 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')

        # Add correlation info
        sig_text = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(0.05, 0.95, f'r = {r:.3f}{sig_text}\np = {p:.2e}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Delta_PCE (%)', fontsize=11)
        ax.set_title(f'{feature.replace("_", " ").title()} vs Delta_PCE', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)

    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Scatter plots saved to: {output_path}")

def create_pair_plot(df, features, target, output_path):
    """Create pair plot matrix colored by Delta_PCE quartile."""
    print("\n" + "=" * 80)
    print("STEP 5: Creating Pair Plot Matrix")
    print("=" * 80)

    # Create quartile bins for Delta_PCE
    df_plot = df.copy()
    df_plot['PCE_Quartile'] = pd.qcut(df_plot[target].dropna(), q=4,
                                       labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

    # Create pair plot
    pair_features = features + [target]
    g = sns.pairplot(df_plot[pair_features + ['PCE_Quartile']],
                     hue='PCE_Quartile',
                     palette=colors[:4],
                     corner=True,
                     diag_kind='kde',
                     plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5},
                     diag_kws={'fill': True, 'alpha': 0.6})

    g.fig.suptitle('Pair Plot: Chemical Descriptors Colored by Delta_PCE Quartile',
                   y=1.02, fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Pair plot saved to: {output_path}")

def create_partial_correlation_network(df, features, target, output_path):
    """Create partial correlation network graph."""
    print("\n" + "=" * 80)
    print("STEP 6: Creating Partial Correlation Network")
    print("=" * 80)

    from scipy.stats import pearsonr
    import networkx as nx

    # Calculate partial correlations
    # Partial correlation between X and Y controlling for all other variables
    n_vars = len(features) + 1
    variables = features + [target]

    # Create correlation matrix
    corr_matrix = df[variables].corr()

    # Calculate precision matrix (inverse of correlation matrix)
    try:
        precision_matrix = np.linalg.inv(corr_matrix.values)

        # Partial correlations from precision matrix
        partial_corr = np.zeros_like(precision_matrix)
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    partial_corr[i, j] = -precision_matrix[i, j] / np.sqrt(precision_matrix[i, i] * precision_matrix[j, j])

        partial_corr_df = pd.DataFrame(partial_corr, index=variables, columns=variables)

        # Create network graph
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create graph
        G = nx.Graph()

        # Add nodes
        for var in variables:
            G.add_node(var)

        # Add edges for significant partial correlations
        threshold = 0.3  # Only show correlations above threshold
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                corr_val = partial_corr_df.iloc[i, j]
                if abs(corr_val) > threshold:
                    G.add_edge(variables[i], variables[j],
                              weight=abs(corr_val),
                              correlation=corr_val)

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw nodes
        node_colors = ['#FF6B6B' if var == target else '#4ECDC4' for var in variables]
        node_sizes = [1000 if var == target else 800 for var in variables]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.8, ax=ax)

        # Draw edges with varying thickness
        edges = G.edges(data=True)
        edge_widths = [e[2]['weight'] * 5 for e in edges]
        edge_colors = ['#FF6B6B' if e[2]['correlation'] < 0 else '#4ECDC4' for e in edges]

        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                              alpha=0.6, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

        # Add edge labels
        edge_labels = {(u, v): f"{d['correlation']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

        ax.set_title('Partial Correlation Network\n(Threshold > |0.3|)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Partial correlation network saved to: {output_path}")

    except np.linalg.LinAlgError:
        print("Warning: Could not compute partial correlations (singular matrix)")
        # Create alternative visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'Partial correlation network could not be computed\n(multicollinearity in data)',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def create_volcano_plot(results_df, output_path):
    """Create volcano plot for correlation significance."""
    print("\n" + "=" * 80)
    print("STEP 7: Creating Volcano Plot")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    x = results_df['Pearson_r']
    y = -np.log10(results_df['Pearson_p'])

    # Color points by significance
    colors_list = []
    for r, p in zip(x, results_df['Pearson_p']):
        if p < 0.001:
            colors_list.append('#C73E1D')  # Strong significance
        elif p < 0.01:
            colors_list.append('#F18F01')  # Moderate significance
        elif p < 0.05:
            colors_list.append('#6A994E')  # Weak significance
        else:
            colors_list.append('#999999')  # Not significant

    # Create scatter plot
    scatter = ax.scatter(x, y, c=colors_list, s=100, alpha=0.7,
                        edgecolors='black', linewidths=1)

    # Add significance threshold lines
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(-np.log10(0.01), color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Add vertical lines at r = 0
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # Add labels
    for i, feature in enumerate(results_df['Feature']):
        ax.annotate(feature.replace('_', ' ').title(),
                   (x[i], y[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)

    ax.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=12)
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    ax.set_title('Volcano Plot: Correlation Significance for Chemical Descriptors',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#C73E1D', label='p < 0.001'),
        Patch(facecolor='#F18F01', label='p < 0.01'),
        Patch(facecolor='#6A994E', label='p < 0.05'),
        Patch(facecolor='#999999', label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Volcano plot saved to: {output_path}")

def create_feature_importance_pca(df, features, target, output_path):
    """Create PCA-based feature importance visualization."""
    print("\n" + "=" * 80)
    print("STEP 8: Creating PCA Feature Importance")
    print("=" * 80)

    # Prepare data
    valid_idx = df[features + [target]].notna().all(axis=1)
    X = df.loc[valid_idx, features]
    y = df.loc[valid_idx, target]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Explained variance ratio
    ax1 = axes[0]
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    ax1.bar(range(1, len(features)+1), pca.explained_variance_ratio_,
           alpha=0.7, color=colors[0], label='Individual')
    ax1.plot(range(1, len(features)+1), cumsum_var,
            marker='o', color=colors[1], linewidth=2, label='Cumulative')
    ax1.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Variance Explained', fontsize=11)
    ax1.set_title('PCA Explained Variance', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Feature loadings
    ax2 = axes[1]
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(loadings,
                                 columns=[f'PC{i}' for i in range(1, len(features)+1)],
                                 index=features)

    # Plot heatmap
    sns.heatmap(loading_matrix.iloc[:, :3], annot=True, fmt='.3f',
               cmap='RdBu_r', center=0, square=True, linewidths=0.5,
               cbar_kws={"label": "Loading"}, ax=ax2)
    ax2.set_title('Feature Loadings on First 3 PCs', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Principal Component', fontsize=11)
    ax2.set_ylabel('Feature', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"PCA feature importance saved to: {output_path}")

def generate_report(df, results_df, corr_matrix, features, output_path):
    """Generate comprehensive correlation analysis report."""
    print("\n" + "=" * 80)
    print("STEP 9: Generating Analysis Report")
    print("=" * 80)

    report = []
    report.append("=" * 80)
    report.append("QSPR CORRELATION ANALYSIS REPORT")
    report.append("Perovskite Solar Cell Modulators")
    report.append("=" * 80)
    report.append(f"\nAnalysis Date: 2026-02-20")
    report.append(f"Dataset: {df.shape[0]} compounds with {len(results_df)} chemical descriptors")
    report.append(f"Target Variable: Delta_PCE (Power Conversion Efficiency Improvement)")

    # Dataset overview
    report.append("\n" + "=" * 80)
    report.append("1. DATASET OVERVIEW")
    report.append("=" * 80)

    report.append(f"\nTotal number of compounds analyzed: {df.shape[0]}")
    report.append(f"Number of unique CAS numbers: {df['cas_number'].nunique()}")
    report.append(f"\nTarget variable (Delta_PCE) statistics:")
    report.append(f"  Mean: {df['Delta_PCE'].mean():.3f}%")
    report.append(f"  Median: {df['Delta_PCE'].median():.3f}%")
    report.append(f"  Std Dev: {df['Delta_PCE'].std():.3f}%")
    report.append(f"  Range: [{df['Delta_PCE'].min():.3f}, {df['Delta_PCE'].max():.3f}]%")

    # Correlation results
    report.append("\n" + "=" * 80)
    report.append("2. CORRELATION ANALYSIS RESULTS")
    report.append("=" * 80)

    report.append("\nRanking by Pearson correlation magnitude:")
    report.append("-" * 80)

    results_sorted = results_df.sort_values('Pearson_r', key=abs, ascending=False)
    for _, row in results_sorted.iterrows():
        report.append(f"\n{row['Feature'].replace('_', ' ').title()}:")
        report.append(f"  Pearson r: {row['Pearson_r']:.4f} (95% CI: [{row['Pearson_CI_lower']:.4f}, {row['Pearson_CI_upper']:.4f}])")
        report.append(f"  p-value: {row['Pearson_p']:.4e}")
        report.append(f"  Adjusted p-value (Bonferroni): {row['Pearson_p_bonf']:.4e}")
        report.append(f"  Spearman ρ: {row['Spearman_rho']:.4f} (p: {row['Spearman_p']:.4e})")
        report.append(f"  Kendall τ: {row['Kendall_tau']:.4f} (p: {row['Kendall_p']:.4e})")

        # Interpretation
        if row['Pearson_p_bonf'] < 0.001:
            sig_level = "highly significant"
        elif row['Pearson_p_bonf'] < 0.01:
            sig_level = "very significant"
        elif row['Pearson_p_bonf'] < 0.05:
            sig_level = "significant"
        else:
            sig_level = "not significant"

        direction = "positive" if row['Pearson_r'] > 0 else "negative"
        strength = "strong" if abs(row['Pearson_r']) > 0.5 else "moderate" if abs(row['Pearson_r']) > 0.3 else "weak"

        report.append(f"  Interpretation: {sig_level} {direction} {strength} correlation")

    # Summary of significant findings
    report.append("\n" + "=" * 80)
    report.append("3. SUMMARY OF SIGNIFICANT FINDINGS")
    report.append("=" * 80)

    significant = results_sorted[results_sorted['Pearson_p_bonf'] < 0.05]
    if len(significant) > 0:
        report.append(f"\n{len(significant)} descriptor(s) show statistically significant correlation with Delta_PCE")
        report.append("after Bonferroni correction for multiple testing:")
        for _, row in significant.iterrows():
            report.append(f"  - {row['Feature'].replace('_', ' ').title()}: r = {row['Pearson_r']:.4f}")
    else:
        report.append("\nNo descriptors show statistically significant correlation with Delta_PCE")
        report.append("after Bonferroni correction.")

    # Inter-correlations
    report.append("\n" + "=" * 80)
    report.append("4. DESCRIPTOR INTER-CORRELATIONS")
    report.append("=" * 80)

    report.append("\nHigh inter-correlations among descriptors (|r| > 0.7):")
    high_corr = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr.append((features[i], features[j], corr_val))

    if high_corr:
        for feat1, feat2, corr_val in high_corr:
            report.append(f"  {feat1} <-> {feat2}: r = {corr_val:.4f}")
        report.append("\nNote: High inter-correlations indicate multicollinearity,")
        report.append("which may affect the interpretation of individual descriptor effects.")
    else:
        report.append("  No high inter-correlations detected.")

    # Recommendations
    report.append("\n" + "=" * 80)
    report.append("5. RECOMMENDATIONS FOR FURTHER ANALYSIS")
    report.append("=" * 80)

    if len(significant) > 0:
        report.append("\nBased on the correlation analysis, the following approaches are recommended:")
        report.append("\n1. Feature Selection:")
        for _, row in significant.iterrows():
            report.append(f"   - Prioritize {row['Feature'].replace('_', ' ').title()} in QSPR models")

        if len(high_corr) > 0:
            report.append("\n2. Multicollinearity Management:")
            report.append("   - Consider dimensionality reduction (PCA)")
            report.append("   - Use regularization techniques (Ridge, Lasso)")
            report.append("   - Apply variance inflation factor (VIF) analysis")
    else:
        report.append("\nGiven the lack of strong linear correlations, consider:")
        report.append("\n1. Non-linear Relationships:")
        report.append("   - Explore polynomial or spline regression")
        report.append("   - Apply random forest or gradient boosting models")
        report.append("   - Consider neural network approaches")

        report.append("\n2. Feature Engineering:")
        report.append("   - Create interaction terms between descriptors")
        report.append("   - Apply logarithmic or power transformations")
        report.append("   - Consider molecular fingerprints or graph-based descriptors")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Write report to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Analysis report saved to: {output_path}")

    # Print report to console
    print('\n'.join(report))

def main():
    """Main analysis workflow."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 20 + "QSPR CORRELATION ANALYSIS" + " " * 20 + "*")
    print("*" + " " * 15 + "Perovskite Solar Cell Modulators" + " " * 15 + "*")
    print("*" * 80)

    # Define paths
    data_path = '/share/yhm/test/AutoML_EDA/processed_data.csv'
    figures_dir = '/share/yhm/test/AutoML_EDA/figures'
    tables_dir = '/share/yhm/test/AutoML_EDA/tables'

    # Step 1: Load and prepare data
    df, features, target = load_and_prepare_data(data_path)

    # Step 2: Calculate correlations
    results_df = calculate_correlations(df, features, target)

    # Save correlation statistics table
    stats_path = f'{tables_dir}/correlation_statistics.csv'
    results_df.to_csv(stats_path, index=False)
    print(f"\nCorrelation statistics saved to: {stats_path}")

    # Step 3: Create correlation heatmap
    corr_matrix = create_correlation_heatmap(df, features, target,
                                             f'{figures_dir}/correlation_heatmap.png')

    # Step 4: Create scatter plots
    create_scatter_plots_with_regression(df, features, target,
                                         f'{figures_dir}/correlation_scatter_plots.png')

    # Step 5: Create pair plot
    create_pair_plot(df, features, target,
                    f'{figures_dir}/correlation_pair_plot.png')

    # Step 6: Create partial correlation network
    create_partial_correlation_network(df, features, target,
                                      f'{figures_dir}/correlation_network.png')

    # Step 7: Create volcano plot
    create_volcano_plot(results_df, f'{figures_dir}/correlation_volcano.png')

    # Step 8: Create PCA feature importance
    create_feature_importance_pca(df, features, target,
                                 f'{figures_dir}/correlation_pca_importance.png')

    # Step 9: Generate comprehensive report
    generate_report(df, results_df, corr_matrix, features,
                   f'{figures_dir}/correlation_report.txt')

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results have been saved to:")
    print(f"  Figures: {figures_dir}/")
    print(f"  Tables: {tables_dir}/")
    print("\nGenerated files:")
    print("  - correlation_heatmap.png")
    print("  - correlation_scatter_plots.png")
    print("  - correlation_pair_plot.png")
    print("  - correlation_network.png")
    print("  - correlation_volcano.png")
    print("  - correlation_pca_importance.png")
    print("  - correlation_statistics.csv")
    print("  - correlation_matrix.csv")
    print("  - correlation_report.txt")
    print("=" * 80)

if __name__ == "__main__":
    main()
