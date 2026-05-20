#!/usr/bin/env python3
"""
Chemical Space Visualization for QSPR Analysis (Optimized)

This script performs comprehensive chemical space analysis with optimized settings
for faster computation on large datasets.
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
print("CHEMICAL SPACE VISUALIZATION FOR QSPR ANALYSIS (OPTIMIZED)")
print("="*80)

# Load data
print("\n1. Loading data...")
data_path = '/share/yhm/test/AutoML_EDA/processed_data.csv'
df = pd.read_csv(data_path)
print(f"   Total samples: {len(df)}")

# Extract chemical descriptors
print("\n2. Extracting chemical descriptors...")
descriptors = ['molecular_weight', 'h_bond_donors', 'h_bond_acceptors',
               'rotatable_bonds', 'tpsa', 'log_p']

available_descriptors = [d for d in descriptors if d in df.columns]
print(f"   Available descriptors: {available_descriptors}")

# Filter out rows with missing descriptor values
df_valid = df.dropna(subset=available_descriptors + ['Delta_PCE']).copy()
print(f"   Valid samples: {len(df_valid)}")

X_desc = df_valid[available_descriptors].values
y = df_valid['Delta_PCE'].values
valid_indices = df_valid.index.tolist()

# Standardize descriptors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_desc)
print(f"   Descriptor matrix shape: {X_scaled.shape}")

# Create Delta_PCE quartiles
y_quartiles = pd.qcut(y, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

# Load fingerprints
print("\n3. Loading fingerprints...")
X_ecfp = None
X_maccs = None

ecfp_path = '/share/yhm/test/AutoML_EDA/fingerprints/ecfp_fingerprints.csv'
maccs_path = '/share/yhm/test/AutoML_EDA/fingerprints/maccs_fingerprints.csv'

if os.path.exists(ecfp_path):
    ecfp_df = pd.read_csv(ecfp_path)
    fingerprint_cols = [col for col in ecfp_df.columns if col.startswith('Bit_')]
    ecfp_df = ecfp_df[fingerprint_cols]
    if len(ecfp_df) == len(df):
        X_ecfp = ecfp_df.iloc[valid_indices].values.astype(np.float32)
        print(f"   ECFP: {X_ecfp.shape}")

if os.path.exists(maccs_path):
    maccs_df = pd.read_csv(maccs_path)
    fingerprint_cols = [col for col in maccs_df.columns if col.startswith('MACCS_') or col.startswith('Bit_')]
    if len(fingerprint_cols) > 0:
        maccs_df = maccs_df[fingerprint_cols]
        if len(maccs_df) == len(df):
            X_maccs = maccs_df.iloc[valid_indices].values.astype(np.float32)
            print(f"   MACCS: {X_maccs.shape}")
    else:
        print("   No MACCS fingerprint columns found")

# ============================================================================
# PCA ANALYSIS
# ============================================================================
print("\n4. PCA Analysis...")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"   PC1: {pca.explained_variance_ratio_[0]*100:.1f}% variance")
print(f"   PC2: {pca.explained_variance_ratio_[1]*100:.1f}% variance")
print(f"   PC3: {pca.explained_variance_ratio_[2]*100:.1f}% variance")

# PCA 2D Plot
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
axes[1].set_title('PCA by Delta_PCE Quartiles', fontsize=14, fontweight='bold')
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
ax.set_title('3D PCA of Chemical Descriptors', fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Delta_PCE (%)', fontsize=11)

ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/chem_space_pca_3d.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: chem_space_pca_3d.png")

# PCA Loadings Plot
fig, ax = plt.subplots(figsize=(10, 8))

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

for i, descriptor in enumerate(available_descriptors):
    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
             head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.6)
    ax.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, descriptor,
            fontsize=10, ha='center', va='center', fontweight='bold')

circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
ax.add_patch(circle)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('PCA Loadings Plot - Chemical Descriptors', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/chem_space_pca_loadings.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: chem_space_pca_loadings.png")

# ============================================================================
# t-SNE ANALYSIS (Optimized - only on descriptors and MACCS)
# ============================================================================
print("\n5. t-SNE Analysis (optimized)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# t-SNE on chemical descriptors (faster)
print("   Running t-SNE on chemical descriptors...")
tsne_desc = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=750)
X_tsne_desc = tsne_desc.fit_transform(X_scaled)

scatter1 = axes[0].scatter(X_tsne_desc[:, 0], X_tsne_desc[:, 1], c=y, cmap='RdYlGn',
                          s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('t-SNE 1', fontsize=11)
axes[0].set_ylabel('t-SNE 2', fontsize=11)
axes[0].set_title('t-SNE: Chemical Descriptors', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Delta_PCE (%)')

# t-SNE on MACCS fingerprints (smaller, faster)
if X_maccs is not None:
    print("   Running t-SNE on MACCS fingerprints...")
    tsne_maccs = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=750)
    X_tsne_maccs = tsne_maccs.fit_transform(X_maccs)

    scatter2 = axes[1].scatter(X_tsne_maccs[:, 0], X_tsne_maccs[:, 1], c=y, cmap='RdYlGn',
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('t-SNE 1', fontsize=11)
    axes[1].set_ylabel('t-SNE 2', fontsize=11)
    axes[1].set_title('t-SNE: MACCS Fingerprints', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Delta_PCE (%)')
else:
    axes[1].text(0.5, 0.5, 'MACCS fingerprints\nnot available',
                ha='center', va='center', fontsize=12, transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/chem_space_tsne.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: chem_space_tsne.png")

# ============================================================================
# UMAP ANALYSIS (Optimized)
# ============================================================================
print("\n6. UMAP Analysis (optimized)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# UMAP on chemical descriptors
print("   Running UMAP on chemical descriptors...")
reducer_desc = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap_desc = reducer_desc.fit_transform(X_scaled)

scatter1 = axes[0].scatter(X_umap_desc[:, 0], X_umap_desc[:, 1], c=y, cmap='RdYlGn',
                          s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('UMAP 1', fontsize=11)
axes[0].set_ylabel('UMAP 2', fontsize=11)
axes[0].set_title('UMAP: Chemical Descriptors', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Delta_PCE (%)')

# UMAP on MACCS fingerprints (faster)
if X_maccs is not None:
    print("   Running UMAP on MACCS fingerprints...")
    reducer_maccs = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap_maccs = reducer_maccs.fit_transform(X_maccs)

    scatter2 = axes[1].scatter(X_umap_maccs[:, 0], X_umap_maccs[:, 1], c=y, cmap='RdYlGn',
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('UMAP 1', fontsize=11)
    axes[1].set_ylabel('UMAP 2', fontsize=11)
    axes[1].set_title('UMAP: MACCS Fingerprints', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Delta_PCE (%)')
else:
    axes[1].text(0.5, 0.5, 'MACCS fingerprints\nnot available',
                ha='center', va='center', fontsize=12, transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig('/share/yhm/test/AutoML_EDA/figures/chem_space_umap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: chem_space_umap.png")

# ============================================================================
# CLUSTERING AND DIVERSITY ANALYSIS
# ============================================================================
print("\n7. Clustering and Diversity Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_scaled)

# DBSCAN clustering
dbscan = DBSCAN(eps=2.0, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_scaled)

# 1. K-means in PCA space
colors_clusters = plt.cm.tab10(np.linspace(0, 1, 4))
for i in range(4):
    mask = clusters_kmeans == i
    axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors_clusters[i]], label=f'Cluster {i+1}',
                      s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[0, 0].set_title('K-means Clustering (k=4)', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 2. DBSCAN in PCA space
unique_labels = set(clusters_dbscan)
colors_dbscan = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
for i, label in enumerate(unique_labels):
    mask = clusters_dbscan == label
    label_name = 'Noise' if label == -1 else f'Cluster {label+1}'
    axes[0, 1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors_dbscan[i]], label=label_name,
                      s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[0, 1].set_title('DBSCAN Clustering', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 3. Distribution density
from scipy.stats import gaussian_kde
xy = np.vstack([X_pca[:, 0], X_pca[:, 1]])
z = gaussian_kde(xy)(xy)
axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=z, cmap='viridis',
                  s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[1, 0].set_title('Chemical Space Distribution Density', fontsize=12, fontweight='bold')
plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Density')

# 4. Cluster vs Delta_PCE
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
print("\n8. Generating report...")

report_path = '/share/yhm/test/AutoML_EDA/chem_space_report.txt'

with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CHEMICAL SPACE ANALYSIS REPORT FOR QSPR STUDY\n")
    f.write("Analysis Date: 2026-02-20\n")
    f.write("="*80 + "\n\n")

    # Data Summary
    f.write("1. DATA SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Valid samples: {len(df_valid)}\n")
    f.write(f"Chemical descriptors: {', '.join(available_descriptors)}\n\n")

    # Delta_PCE Statistics
    f.write("2. DELTA_PCE STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Mean: {np.mean(y):.3f}%\n")
    f.write(f"Median: {np.median(y):.3f}%\n")
    f.write(f"Std Dev: {np.std(y):.3f}%\n")
    f.write(f"Min: {np.min(y):.3f}%\n")
    f.write(f"Max: {np.max(y):.3f}%\n")
    f.write(f"Range: {np.max(y) - np.min(y):.3f}%\n\n")

    # PCA Results
    f.write("3. PCA ANALYSIS RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write("Explained Variance:\n")
    for i in range(min(5, len(pca.explained_variance_ratio_))):
        f.write(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%\n")
    f.write(f"\nCumulative variance:\n")
    f.write(f"  2 PCs: {sum(pca.explained_variance_ratio_[:2])*100:.2f}%\n")
    f.write(f"  3 PCs: {sum(pca.explained_variance_ratio_[:3])*100:.2f}%\n\n")

    # Loadings
    f.write("PCA Loadings (descriptor contributions):\n")
    for i, desc in enumerate(available_descriptors):
        f.write(f"  {desc}: PC1={pca.components_[0,i]:.3f}, PC2={pca.components_[1,i]:.3f}\n")
    f.write("\n")

    # Clustering
    f.write("4. CLUSTERING RESULTS\n")
    f.write("-" * 80 + "\n")
    for i in range(4):
        mask = clusters_kmeans == i
        cluster_pce = y[mask]
        f.write(f"Cluster {i+1}: n={sum(mask)}, "
               f"mean Delta_PCE={np.mean(cluster_pce):.3f}%, "
               f"std={np.std(cluster_pce):.3f}%\n")

    sil_score = silhouette_score(X_scaled, clusters_kmeans)
    f.write(f"\nSilhouette Score: {sil_score:.3f}\n\n")

    # Fingerprints
    if X_ecfp is not None:
        f.write("5. FINGERPRINT SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"ECFP: {X_ecfp.shape[1]} bits\n")
        if X_maccs is not None:
            f.write(f"MACCS: {X_maccs.shape[1]} bits\n")
        f.write("\n")

    # Interpretation
    f.write("6. INTERPRETATION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Key Findings:\n")
    f.write(f"- First 2 PCs explain {sum(pca.explained_variance_ratio_[:2])*100:.1f}% variance\n")
    f.write(f"- Delta_PCE ranges from {np.min(y):.2f}% to {np.max(y):.2f}%\n")
    f.write(f"- K-means identified 4 clusters with silhouette score {sil_score:.3f}\n")
    f.write(f"\nRecommendations:\n")
    f.write(f"- Use top 3-5 PCs as ML features\n")
    f.write(f"- Consider cluster membership as additional feature\n")
    f.write(f"- Explore non-linear relationships with t-SNE/UMAP\n\n")

    f.write("="*80 + "\n")
    f.write("Generated Visualizations:\n")
    f.write("- chem_space_pca_2d.png\n")
    f.write("- chem_space_pca_3d.png\n")
    f.write("- chem_space_pca_loadings.png\n")
    f.write("- chem_space_tsne.png\n")
    f.write("- chem_space_umap.png\n")
    f.write("- chem_space_clusters.png\n")
    f.write("="*80 + "\n")

print(f"   Saved: {report_path}")

print("\n" + "="*80)
print("CHEMICAL SPACE VISUALIZATION COMPLETE!")
print("="*80)
print("\nAll files saved to:")
print("  Figures: /share/yhm/test/AutoML_EDA/figures/")
print("  Report: /share/yhm/test/AutoML_EDA/chem_space_report.txt")
