#!/usr/bin/env python3
"""
Extended RDKit Molecular Descriptors Calculation for QSPR Analysis
of Perovskite Solar Cell Modulators - Optimized Version
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# File paths
INPUT_FILE = '/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx'
OUTPUT_DIR = '/share/yhm/test/AutoML_EDA'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'

# Descriptor names
DESCRIPTOR_NAMES = ['C', 'H', 'N', 'F', 'O', 'MW', 'LogP', 'TPSA',
                    'H_acceptor', 'H_donor', 'RB', 'Aromatic_rings',
                    'Aliphatic_rings', 'Saturated_rings', 'Heteroatoms', 'QED', 'IPC']

def calculate_extended_descriptors(smiles):
    """Calculate 17 RDKit molecular descriptors from SMILES string."""
    if pd.isna(smiles) or smiles == '':
        return {name: np.nan for name in DESCRIPTOR_NAMES}

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {name: np.nan for name in DESCRIPTOR_NAMES}

    # Atom counts
    atom_counts = {'C': 0, 'H': 0, 'N': 0, 'F': 0, 'O': 0}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atom_counts:
            atom_counts[symbol] += 1

    # Add hydrogen count explicitly
    atom_counts['H'] = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())

    # IPC calculation
    try:
        ipc = Descriptors.Ipc(mol)
    except:
        ipc = np.nan

    descriptors = {
        'C': atom_counts['C'],
        'H': atom_counts['H'],
        'N': atom_counts['N'],
        'F': atom_counts['F'],
        'O': atom_counts['O'],
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'H_acceptor': Descriptors.NumHAcceptors(mol),
        'H_donor': Descriptors.NumHDonors(mol),
        'RB': Descriptors.NumRotatableBonds(mol),
        'Aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'Aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
        'Saturated_rings': rdMolDescriptors.CalcNumSaturatedRings(mol),
        'Heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'QED': qed(mol),
        'IPC': ipc
    }

    return descriptors

def main():
    print("="*70)
    print("Extended RDKit Molecular Descriptors Calculation for QSPR Analysis")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_excel(INPUT_FILE)
    print(f"   Loaded {len(df)} rows")

    # Extract SMILES and calculate descriptors
    print("\n2. Calculating 17 RDKit molecular descriptors...")
    smiles_series = df['smiles'].fillna('')

    # Vectorized calculation
    desc_list = [calculate_extended_descriptors(s) for s in smiles_series]
    desc_df = pd.DataFrame(desc_list)

    # Add descriptors to original dataframe
    for col in DESCRIPTOR_NAMES:
        df[col] = desc_df[col]

    # Calculate Delta_PCE
    print("\n3. Calculating Delta_PCE...")
    if 'jv_reverse_scan_pce' in df.columns and 'jv_reverse_scan_pce_without_modulator' in df.columns:
        df['Delta_PCE'] = df['jv_reverse_scan_pce'] - df['jv_reverse_scan_pce_without_modulator']
        print(f"   Delta_PCE calculated successfully")
        valid_delta = df['Delta_PCE'].dropna()
        if len(valid_delta) > 0:
            print(f"   Delta_PCE range: [{valid_delta.min():.4f}, {valid_delta.max():.4f}]")
    else:
        print("   Warning: Required columns for Delta_PCE calculation not found!")
        df['Delta_PCE'] = np.nan

    # Save extended descriptors
    print("\n4. Saving results...")
    output_csv = f'{OUTPUT_DIR}/rdkit_extended_descriptors.csv'
    df.to_csv(output_csv, index=False)
    print(f"   Saved extended descriptors to: {output_csv}")

    # Correlation analysis
    print("\n5. Performing correlation analysis with Delta_PCE...")
    valid_data = df[DESCRIPTOR_NAMES + ['Delta_PCE']].dropna()

    correlations = {}
    for col in DESCRIPTOR_NAMES:
        if valid_data[col].std() > 0:
            corr = valid_data[col].corr(valid_data['Delta_PCE'])
            correlations[col] = corr
        else:
            correlations[col] = np.nan

    corr_df = pd.DataFrame({
        'Descriptor': list(correlations.keys()),
        'Correlation_with_Delta_PCE': list(correlations.values())
    }).sort_values('Correlation_with_Delta_PCE', key=abs, ascending=False)

    # Full correlation matrix
    corr_matrix = df[DESCRIPTOR_NAMES + ['Delta_PCE']].corr()

    output_corr = f'{OUTPUT_DIR}/rdkit_correlation_analysis.csv'
    corr_df.to_csv(output_corr, index=False)
    print(f"   Saved correlation analysis to: {output_corr}")

    # Print correlation summary
    print("\n   Top correlations with Delta_PCE:")
    for _, row in corr_df.head(10).iterrows():
        print(f"      {row['Descriptor']}: {row['Correlation_with_Delta_PCE']:.4f}")

    # Summary statistics
    print("\n6. Generating summary statistics...")
    summary_stats = df[DESCRIPTOR_NAMES].describe()

    with open(f'{OUTPUT_DIR}/rdkit_analysis_summary.txt', 'w') as f:
        f.write("Extended RDKit Molecular Descriptors Analysis Summary\n")
        f.write("="*70 + "\n\n")

        f.write("1. Dataset Overview\n")
        f.write("-"*40 + "\n")
        f.write(f"Total molecules: {len(df)}\n")
        f.write(f"Valid SMILES processed: {df['smiles'].notna().sum()}\n")
        f.write(f"Valid Delta_PCE values: {df['Delta_PCE'].notna().sum()}\n\n")

        f.write("2. Descriptor Statistics\n")
        f.write("-"*40 + "\n")
        f.write(summary_stats.to_string())
        f.write("\n\n")

        f.write("3. Correlation with Delta_PCE\n")
        f.write("-"*40 + "\n")
        f.write(corr_df.to_string(index=False))
        f.write("\n\n")

        f.write("4. Interpretation\n")
        f.write("-"*40 + "\n")
        if len(corr_df) > 0 and not pd.isna(corr_df.iloc[0]['Correlation_with_Delta_PCE']):
            top_corr = corr_df.iloc[0]
            f.write(f"Strongest correlation: {top_corr['Descriptor']} (r={top_corr['Correlation_with_Delta_PCE']:.4f})\n")
        f.write("Positive correlations indicate features associated with PCE improvement.\n")
        f.write("Negative correlations indicate features associated with PCE degradation.\n")

    print(f"   Saved summary to: {OUTPUT_DIR}/rdkit_analysis_summary.txt")

    # Create visualizations
    print("\n7. Creating visualizations...")
    create_visualizations(df, corr_df, corr_matrix)

    print("\n" + "="*70)
    print("Calculation complete!")
    print("="*70)

    return df, corr_df, corr_matrix

def create_visualizations(df, corr_df, corr_matrix):
    """Create all required visualizations."""
    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Correlation Heatmap
    print("   Creating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
    ax.set_title('RDKit Descriptors Correlation Matrix with Delta_PCE', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/rdkit_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Scatter plots for top 5 correlated features
    print("   Creating scatter plots for top 5 correlated features...")
    top_5_features = corr_df.head(5)['Descriptor'].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(top_5_features):
        if idx < len(axes):
            valid_data = df[[feature, 'Delta_PCE']].dropna()
            if len(valid_data) > 0:
                ax = axes[idx]
                ax.scatter(valid_data[feature], valid_data['Delta_PCE'],
                          alpha=0.5, edgecolors='white', linewidth=0.5)

                # Add trend line
                z = np.polyfit(valid_data[feature], valid_data['Delta_PCE'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_data[feature].min(), valid_data[feature].max(), 100)
                ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')

                corr_val = corr_df[corr_df['Descriptor'] == feature]['Correlation_with_Delta_PCE'].values[0]
                ax.set_xlabel(feature, fontsize=11)
                ax.set_ylabel('Delta_PCE', fontsize=11)
                ax.set_title(f'{feature} vs Delta_PCE (r={corr_val:.3f})', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(top_5_features), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Top 5 Correlated Features vs Delta_PCE', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/rdkit_scatter_top5.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Distribution plots for each descriptor
    print("   Creating distribution plots...")
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    axes = axes.flatten()

    for idx, desc in enumerate(DESCRIPTOR_NAMES):
        if idx < len(axes):
            ax = axes[idx]
            valid_data = df[desc].dropna()
            if len(valid_data) > 0:
                ax.hist(valid_data, bins=30, edgecolor='white', alpha=0.7, color='steelblue')
                ax.set_xlabel(desc, fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title(f'{desc} Distribution', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)

    # Add Delta_PCE distribution
    if len(axes) > len(DESCRIPTOR_NAMES):
        ax = axes[len(DESCRIPTOR_NAMES)]
        valid_delta = df['Delta_PCE'].dropna()
        if len(valid_delta) > 0:
            ax.hist(valid_delta, bins=30, edgecolor='white', alpha=0.7, color='darkred')
            ax.set_xlabel('Delta_PCE', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title('Delta_PCE Distribution', fontsize=11, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Zero')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(DESCRIPTOR_NAMES) + 1, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Distribution of RDKit Molecular Descriptors', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/rdkit_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Bar plot of correlations
    print("   Creating correlation bar plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['green' if x > 0 else 'red' for x in corr_df['Correlation_with_Delta_PCE']]
    bars = ax.barh(corr_df['Descriptor'], corr_df['Correlation_with_Delta_PCE'], color=colors, edgecolor='black')
    ax.set_xlabel('Correlation with Delta_PCE', fontsize=12)
    ax.set_ylabel('Descriptor', fontsize=12)
    ax.set_title('RDKit Descriptors: Correlation with Delta_PCE', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, val in zip(bars, corr_df['Correlation_with_Delta_PCE']):
        if not pd.isna(val):
            ax.text(val + 0.01 if val >= 0 else val - 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/rdkit_correlation_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   All visualizations saved successfully!")

if __name__ == "__main__":
    df, corr_df, corr_matrix = main()
