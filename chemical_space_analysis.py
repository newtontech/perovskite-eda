#!/usr/bin/env python3
"""
Chemical Space Visualization for QSPR Analysis

This script performs comprehensive chemical space analysis including:
- PCA analysis (2D/3D)
- t-SNE visualization
- UMAP visualization
- Chemical space diversity analysis
- Cluster identification
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import umap
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Create output directories
os.makedirs('/share/yhm/test/AutoML_EDA/figures', exist_ok=True)

print("="*80)
print("CHEMICAL SPACE VISUALIZATION FOR QSPR ANALYSIS")
print("="*80)

# Load data
print("\n1. Loading data...")
data_path = '/share/yhm/test/AutoML_EDA/processed_data.csv'
df = pd.read_csv(data_path)
print(f"   Total samples: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# Check for Delta_PCE column
if 'Delta_PCE' not in df.columns:
    # Calculate Delta_PCE if not present
    if 'jv_reverse_scan_pce' in df.columns and 'jv_reverse_scan_pce_without_modulator' in df.columns:
        df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']
        print(f"   Created Delta_PCE column (PCE with - PCE without modulator)")

# Extract chemical descriptors
print("\n2. Extracting chemical descriptors...")
descriptors = ['molecular_weight', 'h_bond_donors', 'h_bond_acceptors',
               'rotatable_bonds', 'tpsa', 'log_p']

# Check which descriptors are available
available_descriptors = [d for d in descriptors if d in df.columns]
print(f"   Available descriptors: {available_descriptors}")

# Filter out rows with missing descriptor values
df_valid = df.dropna(subset=available_descriptors + ['Delta_PCE']).copy()
print(f"   Valid samples after filtering: {len(df_valid)}")

# Get valid indices
valid_indices = df_valid.index.tolist()

X_desc = df_valid[available_descriptors].values
y = df_valid['Delta_PCE'].values

# Standardize descriptors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_desc)
print(f"   Descriptor matrix shape: {X_scaled.shape}")

# Load fingerprints if available
print("\n3. Loading fingerprints...")
ecfp_path = '/share/yhm/test/AutoML_EDA/fingerprints/ecfp_fingerprints.csv'
maccs_path = '/share/yhm/test/AutoML_EDA/fingerprints/maccs_fingerprints.csv'
krfp_path = '/share/yhm/test/AutoML_EDA/fingerprints/krfp_fingerprints.npy'

X_ecfp = None
X_maccs = None
X_krfp = None

if os.path.exists(ecfp_path):
    ecfp_df = pd.read_csv(ecfp_path)
    # Remove non-fingerprint columns (SMILES, Delta_PCE, etc.)
    fingerprint_cols = [col for col in ecfp_df.columns if col.startswith('Bit_')]
    ecfp_df = ecfp_df[fingerprint_cols]
    # Filter to match valid samples - use integer indexing
    if len(ecfp_df) == len(df):
        X_ecfp = ecfp_df.iloc[valid_indices].values.astype(np.float32)
        print(f"   ECFP fingerprints loaded: {X_ecfp.shape}")
    else:
        print(f"   Warning: ECFP file length ({len(ecfp_df)}) doesn't match data length ({len(df)})")

if os.path.exists(maccs_path):
    maccs_df = pd.read_csv(maccs_path)
    # Remove non-fingerprint columns (SMILES, Delta_PCE, etc.)
    fingerprint_cols = [col for col in maccs_df.columns if col.startswith('Bit_')]
    maccs_df = maccs_df[fingerprint_cols]
    # Filter to match valid samples - use integer indexing
    if len(maccs_df) == len(df):
        X_maccs = maccs_df.iloc[valid_indices].values.astype(np.float32)
        print(f"   MACCS fingerprints loaded: {X_maccs.shape}")
    else:
        print(f"   Warning: MACCS file length ({len(maccs_df)}) doesn't match data length ({len(df)})")

if os.path.exists(krfp_path):
    X_krfp = np.load(krfp_path)
    # Adjust size if needed - use integer indexing
    if len(X_krfp) == len(df):
        X_krfp = X_krfp[valid_indices]
        print(f"   KRFP fingerprints loaded: {X_krfp.shape}")
    elif len(X_krfp) >= len(df_valid):
        X_krfp = X_krfp[:len(df_valid)]
        print(f"   KRFP fingerprints loaded (truncated): {X_krfp.shape}")
    else:
        print(f"   Warning: KRFP file length ({len(X_krfp)}) is less than valid samples ({len(df_valid)})")

# ============================================================================
# PCA ANALYSIS
# ============================================================================
print("\n4. PCA Analysis...")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"   Explained variance ratio (first 5 PCs): {pca.explained_variance_ratio_[:5]}")
print(f"   Cumulative variance (first 2 PCs): {sum(pca.explained_variance_ratio_[:2]):.3f}")
print(f"   Cumulative variance (first 3 PCs): {sum(pca.explained_variance_ratio_[:3]):.3f}")

# Create quartiles for coloring
y_quartiles = pd.qcut(y, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

# PCA 2D Plot - Continuous Delta_PCE
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Continuous coloring
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn',
                          s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
axes[0].set_title('PCA of Chemical Descriptors\nColored by Delta_PCE', fontsize=14, fontweight='bold')
cbar1 = plt.colorbar(scatter1, ax=axes[0])
cbar1.set_label('Delta_PCE (%)', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Right: Quartile coloring
colors = ['#d73027', '#fee08b', '#1a9850']
for i, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    if quartile in y_quartiles.categories:
        mask = y_quartiles == quartile
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors[i % len(colors)], label=f'{quartile}',
                       s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
axes[1].set_title('PCA of Chemical Descriptors\nColored by Delta_PCE Quartiles', fontsize=14, fontweight='bold')
axes[1].legend(title='Quartile', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/chem_space_pca_2d.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: chem_space_pca_2d.png")

# PCA 3D Plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                    c=y, cmap='RdYlGn', s=50, alpha=0.7,
                    edgecolors='black', linewidth=0.5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=11)
ax.set_title('3D PCA of Chemical Descriptors\nColored by Delta_PCE', fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Delta_PCE (%)', fontsize=11)

ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/chem_space_pca_3d.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: chem_space_pca_3d.png")

# PCA Loadings Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Scale loadings for visibility
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

for i, descriptor in enumerate(available_descriptors):
    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
             head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.6)
    ax.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, descriptor,
            fontsize=10, ha='center', va='center', fontweight='bold')

# Add unit circle
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
ax.add_patch(circle)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('PCA Loadings Plot - Chemical Descriptors', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/chem_space_pca_loadings.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: chem_space_pca_loadings.png")

# ============================================================================
# t-SNE ANALYSIS
# ============================================================================
print("\n5. t-SNE Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# t-SNE on chemical descriptors
print("   Running t-SNE on chemical descriptors...")
tsne_desc = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_valid)//4))
X_tsne_desc = tsne_desc.fit_transform(X_scaled)

# t-SNE on ECFP fingerprints
if X_ecfp is not None:
    print("   Running t-SNE on ECFP fingerprints...")
    tsne_ecfp = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_valid)//4))
    X_tsne_ecfp = tsne_ecfp.fit_transform(X_ecfp)

# t-SNE on MACCS fingerprints
if X_maccs is not None:
    print("   Running t-SNE on MACCS fingerprints...")
    tsne_maccs = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_valid)//4))
    X_tsne_maccs = tsne_maccs.fit_transform(X_maccs)

# Plot t-SNE results
# 1. t-SNE on descriptors
scatter1 = axes[0].scatter(X_tsne_desc[:, 0], X_tsne_desc[:, 1], c=y, cmap='RdYlGn',
                          s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('t-SNE 1', fontsize=11)
axes[0].set_ylabel('t-SNE 2', fontsize=11)
axes[0].set_title('t-SNE: Chemical Descriptors', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Delta_PCE (%)')

# 2. t-SNE on ECFP
if X_ecfp is not None:
    scatter2 = axes[1].scatter(X_tsne_ecfp[:, 0], X_tsne_ecfp[:, 1], c=y, cmap='RdYlGn',
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('t-SNE 1', fontsize=11)
    axes[1].set_ylabel('t-SNE 2', fontsize=11)
    axes[1].set_title('t-SNE: ECFP Fingerprints', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Delta_PCE (%)')
else:
    axes[1].text(0.5, 0.5, 'ECFP fingerprints\nnot available',
                ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

# 3. t-SNE on MACCS
if X_maccs is not None:
    scatter3 = axes[2].scatter(X_tsne_maccs[:, 0], X_tsne_maccs[:, 1], c=y, cmap='RdYlGn',
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[2].set_xlabel('t-SNE 1', fontsize=11)
    axes[2].set_ylabel('t-SNE 2', fontsize=11)
    axes[2].set_title('t-SNE: MACCS Fingerprints', fontsize=12, fontweight='bold')
    plt.colorbar(scatter3, ax=axes[2], label='Delta_PCE (%)')
else:
    axes[2].text(0.5, 0.5, 'MACCS fingerprints\nnot available',
                ha='center', va='center', fontsize=12, transform=axes[2].transAxes)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

# 4. t-SNE on descriptors with quartiles
colors_quartiles = ['#d73027', '#fee08b', '#1a9850']
for i, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    if quartile in y_quartiles.categories:
        mask = y_quartiles == quartile
        axes[3].scatter(X_tsne_desc[mask, 0], X_tsne_desc[mask, 1],
                       c=colors_quartiles[i % len(colors_quartiles)], label=f'{quartile}',
                       s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[3].set_xlabel('t-SNE 1', fontsize=11)
axes[3].set_ylabel('t-SNE 2', fontsize=11)
axes[3].set_title('t-SNE: Descriptors by Quartiles', fontsize=12, fontweight='bold')
axes[3].legend(title='Quartile', fontsize=9)

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/chem_space_tsne.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: chem_space_tsne.png")

# ============================================================================
# UMAP ANALYSIS
# ============================================================================
print("\n6. UMAP Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# UMAP on chemical descriptors
print("   Running UMAP on chemical descriptors...")
reducer_desc = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(df_valid)//10))
X_umap_desc = reducer_desc.fit_transform(X_scaled)

# UMAP on ECFP
X_umap_ecfp = None
if X_ecfp is not None:
    print("   Running UMAP on ECFP fingerprints...")
    reducer_ecfp = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(df_valid)//10))
    X_umap_ecfp = reducer_ecfp.fit_transform(X_ecfp)

# UMAP on MACCS
X_umap_maccs = None
if X_maccs is not None:
    print("   Running UMAP on MACCS fingerprints...")
    reducer_maccs = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(df_valid)//10))
    X_umap_maccs = reducer_maccs.fit_transform(X_maccs)

# Plot UMAP results
# 1. UMAP on descriptors
scatter1 = axes[0].scatter(X_umap_desc[:, 0], X_umap_desc[:, 1], c=y, cmap='RdYlGn',
                          s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('UMAP 1', fontsize=11)
axes[0].set_ylabel('UMAP 2', fontsize=11)
axes[0].set_title('UMAP: Chemical Descriptors', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Delta_PCE (%)')

# 2. UMAP on ECFP
if X_umap_ecfp is not None:
    scatter2 = axes[1].scatter(X_umap_ecfp[:, 0], X_umap_ecfp[:, 1], c=y, cmap='RdYlGn',
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('UMAP 1', fontsize=11)
    axes[1].set_ylabel('UMAP 2', fontsize=11)
    axes[1].set_title('UMAP: ECFP Fingerprints', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Delta_PCE (%)')
else:
    axes[1].text(0.5, 0.5, 'ECFP fingerprints\nnot available',
                ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

# 3. t-SNE vs UMAP comparison
axes[2].scatter(X_tsne_desc[:, 0], X_tsne_desc[:, 1], c='blue', alpha=0.3, s=30, label='t-SNE')
axes[2].scatter(X_umap_desc[:, 0], X_umap_desc[:, 1], c='red', alpha=0.3, s=30, label='UMAP')
axes[2].set_xlabel('Component 1', fontsize=11)
axes[2].set_ylabel('Component 2', fontsize=11)
axes[2].set_title('t-SNE vs UMAP Comparison', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)

# 4. UMAP with quartiles
colors_quartiles = ['#d73027', '#fee08b', '#1a9850']
for i, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    if quartile in y_quartiles.categories:
        mask = y_quartiles == quartile
        axes[3].scatter(X_umap_desc[mask, 0], X_umap_desc[mask, 1],
                       c=colors_quartiles[i % len(colors_quartiles)], label=f'{quartile}',
                       s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[3].set_xlabel('UMAP 1', fontsize=11)
axes[3].set_ylabel('UMAP 2', fontsize=11)
axes[3].set_title('UMAP: Descriptors by Quartiles', fontsize=12, fontweight='bold')
axes[3].legend(title='Quartile', fontsize=9)

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/chem_space_umap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: chem_space_umap.png")

# ============================================================================
# CHEMICAL SPACE DIVERSITY AND CLUSTERING
# ============================================================================
print("\n7. Chemical Space Diversity and Clustering Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# K-means clustering
print("   Performing K-means clustering...")
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_scaled)

# DBSCAN clustering
print("   Performing DBSCAN clustering...")
dbscan = DBSCAN(eps=2.0, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_scaled)

# 1. K-means clusters in PCA space
colors_clusters = plt.cm.tab10(np.linspace(0, 1, n_clusters))
for i in range(n_clusters):
    mask = clusters_kmeans == i
    axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors_clusters[i]], label=f'Cluster {i+1}',
                      s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[0, 0].set_title('K-means Clustering (k=4)', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 2. DBSCAN clusters in PCA space
unique_labels = set(clusters_dbscan)
colors_dbscan = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
for i, label in enumerate(unique_labels):
    mask = clusters_dbscan == label
    if label == -1:
        label_name = 'Noise'
    else:
        label_name = f'Cluster {label+1}'
    axes[0, 1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors_dbscan[i]], label=label_name,
                      s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[0, 1].set_title('DBSCAN Clustering', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 3. Distribution density plot
from scipy.stats import gaussian_kde
xy = np.vstack([X_pca[:, 0], X_pca[:, 1]])
z = gaussian_kde(xy)(xy)
axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=z, cmap='viridis',
                  s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[1, 0].set_title('Chemical Space Distribution Density', fontsize=12, fontweight='bold')
plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Density')

# 4. Cluster analysis with Delta_PCE boxplot
cluster_df = pd.DataFrame({'Cluster': clusters_kmeans, 'Delta_PCE': y})
sns.boxplot(data=cluster_df, x='Cluster', y='Delta_PCE', ax=axes[1, 1], palette='tab10')
axes[1, 1].set_xlabel('K-means Cluster', fontsize=11)
axes[1, 1].set_ylabel('Delta_PCE (%)', fontsize=11)
axes[1, 1].set_title('Delta_PCE Distribution by Cluster', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/chem_space_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: chem_space_clusters.png")

# ============================================================================
# COMPREHENSIVE REPORT
# ============================================================================
print("\n8. Generating comprehensive report...")

report_path = '/share/yhm/test/AutoML_EDA/chem_space_report.txt'

with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CHEMICAL SPACE ANALYSIS REPORT FOR QSPR STUDY\n")
    f.write("Analysis Date: 2026-02-20\n")
    f.write("="*80 + "\n\n")

    # Data Summary
    f.write("1. DATA SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total samples in dataset: {len(df)}\n")
    f.write(f"Valid samples for analysis: {len(df_valid)}\n")
    f.write(f"Samples with missing values: {len(df) - len(df_valid)}\n\n")
    f.write(f"Chemical descriptors analyzed:\n")
    for i, desc in enumerate(available_descriptors, 1):
        f.write(f"  {i}. {desc}\n")
    f.write("\n")

    # Delta_PCE Statistics
    f.write("2. DELTA_PCE STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Mean: {np.mean(y):.3f} %\n")
    f.write(f"Median: {np.median(y):.3f} %\n")
    f.write(f"Std Dev: {np.std(y):.3f} %\n")
    f.write(f"Min: {np.min(y):.3f} %\n")
    f.write(f"Max: {np.max(y):.3f} %\n")
    f.write(f"Range: {np.max(y) - np.min(y):.3f} %\n\n")
    f.write("Quartile Statistics:\n")
    for i, quartile in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        if quartile in y_quartiles.categories:
            mask = y_quartiles == quartile
            f.write(f"  {quartile}: n={sum(mask)}, mean={np.mean(y[mask]):.3f} %, "
                   f"std={np.std(y[mask]):.3f} %\n")
    f.write("\n")

    # PCA Results
    f.write("3. PCA ANALYSIS RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write("Explained Variance Ratio:\n")
    for i in range(min(6, len(pca.explained_variance_ratio_))):
        f.write(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%\n")
    f.write(f"\nCumulative Variance:\n")
    f.write(f"  First 2 PCs: {sum(pca.explained_variance_ratio_[:2])*100:.2f}%\n")
    f.write(f"  First 3 PCs: {sum(pca.explained_variance_ratio_[:3])*100:.2f}%\n")
    f.write(f"  First 6 PCs: {sum(pca.explained_variance_ratio_[:6])*100:.2f}%\n\n")

    f.write("PCA Loadings (descriptor contributions):\n")
    loadings_df = pd.DataFrame(
        pca.components_.T[:, :3],
        columns=['PC1', 'PC2', 'PC3'],
        index=available_descriptors
    )
    f.write(loadings_df.to_string())
    f.write("\n\n")

    # Chemical Diversity Metrics
    f.write("4. CHEMICAL DIVERSITY METRICS\n")
    f.write("-" * 80 + "\n")

    # Calculate diversity metrics
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))
    mean_distance = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
    f.write(f"Mean pairwise Euclidean distance: {mean_distance:.4f}\n")

    # Descriptor ranges
    f.write("\nDescriptor Ranges (min-max):\n")
    for desc in available_descriptors:
        min_val = df_valid[desc].min()
        max_val = df_valid[desc].max()
        f.write(f"  {desc}: {min_val:.3f} - {max_val:.3f}\n")
    f.write("\n")

    # Clustering Results
    f.write("5. CLUSTERING ANALYSIS RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write("K-means Clustering (k=4):\n")
    for i in range(n_clusters):
        mask = clusters_kmeans == i
        cluster_pce = y[mask]
        f.write(f"  Cluster {i+1}: n={sum(mask)}, "
               f"mean Delta_PCE={np.mean(cluster_pce):.3f}%, "
               f"std={np.std(cluster_pce):.3f}%\n")

    # Silhouette score
    sil_score = silhouette_score(X_scaled, clusters_kmeans)
    f.write(f"\nSilhouette Score (K-means): {sil_score:.3f}\n\n")

    # DBSCAN results
    n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
    n_noise = list(clusters_dbscan).count(-1)
    f.write("DBSCAN Clustering:\n")
    f.write(f"  Number of clusters: {n_clusters_dbscan}\n")
    f.write(f"  Number of noise points: {n_noise}\n\n")

    # Fingerprint Analysis
    if X_ecfp is not None:
        f.write("6. FINGERPRINT ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"ECFP: {X_ecfp.shape[1]} bits, "
               f"mean density={np.mean(X_ecfp):.4f}, "
               f"sparsity={1 - np.mean(X_ecfp > 0):.4f}\n")
    if X_maccs is not None:
        f.write(f"MACCS: {X_maccs.shape[1]} bits, "
               f"mean density={np.mean(X_maccs):.4f}, "
               f"sparsity={1 - np.mean(X_maccs > 0):.4f}\n")
    if X_krfp is not None:
        f.write(f"KRFP: {X_krfp.shape[1]} bits\n")
    f.write("\n")

    # Interpretation
    f.write("7. INTERPRETATION AND RECOMMENDATIONS\n")
    f.write("-" * 80 + "\n")
    f.write("Key Findings:\n")
    f.write(f"1. The first two principal components explain {sum(pca.explained_variance_ratio_[:2])*100:.1f}% "
           f"of the variance in chemical descriptor space.\n")
    f.write(f"2. Three principal components explain {sum(pca.explained_variance_ratio_[:3])*100:.1f}% "
           f"of the variance, suggesting a moderately complex chemical space.\n")
    f.write(f"3. Delta_PCE ranges from {np.min(y):.2f}% to {np.max(y):.2f}% with "
           f"moderate variability (std={np.std(y):.2f}%).\n")
    f.write(f"4. K-means clustering identified {n_clusters} distinct chemical regions "
           f"with silhouette score of {sil_score:.3f}.\n\n")

    f.write("Recommendations for QSPR Modeling:\n")
    f.write("- Consider using the top 3-5 PCs as features for ML models\n")
    f.write("- Investigate the relationship between descriptor loadings and Delta_PCE\n")
    f.write("- Use cluster membership as additional categorical features\n")
    f.write("- Consider non-linear transformations given the moderate variance explained\n")
    f.write("- Explore molecular fingerprints for capturing substructure effects\n\n")

    f.write("Generated Visualizations:\n")
    f.write("1. chem_space_pca_2d.png - 2D PCA plots (continuous and quartile coloring)\n")
    f.write("2. chem_space_pca_3d.png - 3D PCA plot\n")
    f.write("3. chem_space_pca_loadings.png - Descriptor loadings plot\n")
    f.write("4. chem_space_tsne.png - t-SNE visualizations on descriptors and fingerprints\n")
    f.write("5. chem_space_umap.png - UMAP visualizations and t-SNE comparison\n")
    f.write("6. chem_space_clusters.png - Clustering analysis and density plots\n\n")

    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"   Saved: {report_path}")

print("\n" + "="*80)
print("CHEMICAL SPACE VISUALIZATION COMPLETE!")
print("="*80)
print("\nAll visualizations saved to: /share/yhm/test/AutoML_EDA/figures/")
print("Analysis report saved to: /share/yhm/test/AutoML_EDA/chem_space_report.txt")
print("\nGenerated files:")
for file in ['chem_space_pca_2d.png', 'chem_space_pca_3d.png', 'chem_space_pca_loadings.png',
             'chem_space_tsne.png', 'chem_space_umap.png', 'chem_space_clusters.png',
             'chem_space_report.txt']:
    print(f"  - {file}")
