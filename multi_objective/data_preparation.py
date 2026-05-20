"""
Multi-Objective Data Preparation for Perovskite Solar Cells
Research: Physics-Informed Multi-Output Learning

Author: OpenClaw AI Assistant
Date: 2026-03-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


class MultiObjectiveDataPreparation:
    """
    Prepare multi-target data for physics-informed machine learning.
    
    Targets:
    - PCE: Power Conversion Efficiency (%)
    - Voc: Open Circuit Voltage (V)
    - Jsc: Short Circuit Current Density (mA/cm²)
    - FF: Fill Factor (dimensionless, 0-1)
    
    Physical constraint: PCE = (Voc × Jsc × FF) / Pin
    """
    
    # Physical limits for data validation
    PHYSICAL_LIMITS = {
        'PCE': {'min': 0, 'max': 35},      # Max ~30-35% for perovskites
        'Voc': {'min': 0.5, 'max': 1.5},   # Typical range 0.5-1.5 V
        'Jsc': {'min': 0, 'max': 30},      # Typical range 0-30 mA/cm²
        'FF': {'min': 0.3, 'max': 0.9}     # Typical range 0.3-0.9
    }
    
    def __init__(self, input_path: str, output_dir: str = 'multi_objective/data'):
        """
        Initialize data preparation.
        
        Args:
            input_path: Path to cleaned perovskite data
            output_dir: Directory to save processed data
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.df_clean = None
        self.stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and display basic data information."""
        print("=" * 60)
        print("Multi-Objective Data Preparation")
        print("=" * 60)
        
        self.df = pd.read_csv(self.input_path)
        print(f"\n📊 Original data: {len(self.df):,} samples, {len(self.df.columns)} features")
        
        # Check target columns
        target_cols = ['PCE', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF']
        
        available_targets = []
        for col in target_cols:
            # Try both original and cleaned column names
            if col in self.df.columns:
                available_targets.append(col)
            elif col.replace('JV_default_', '') in self.df.columns:
                available_targets.append(col.replace('JV_default_', ''))
        
        print(f"\n🎯 Available targets: {available_targets}")
        
        return self.df
    
    def extract_targets(self) -> pd.DataFrame:
        """
        Extract and rename target variables.
        
        Returns:
            DataFrame with standardized target columns
        """
        df_targets = self.df.copy()
        
        # Map column names
        target_mapping = {
            'PCE': 'PCE',
            'JV_default_Voc': 'Voc',
            'JV_default_Jsc': 'Jsc',
            'JV_default_FF': 'FF',
            'Voc': 'Voc',
            'Jsc': 'Jsc',
            'FF': 'FF'
        }
        
        # Rename columns
        rename_dict = {}
        for old, new in target_mapping.items():
            if old in df_targets.columns and new not in df_targets.columns:
                rename_dict[old] = new
        
        df_targets = df_targets.rename(columns=rename_dict)
        
        # Select target columns
        target_cols = ['PCE', 'Voc', 'Jsc', 'FF']
        available = [col for col in target_cols if col in df_targets.columns]
        
        self.df = df_targets[available].copy()
        print(f"\n✅ Extracted {len(available)} target columns: {available}")
        
        return self.df
    
    def clean_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean multi-target data by removing invalid entries.
        
        Returns:
            Tuple of (cleaned DataFrame, cleaning statistics)
        """
        df = self.df.copy()
        initial_count = len(df)
        
        print("\n" + "=" * 60)
        print("Data Cleaning")
        print("=" * 60)
        
        # Remove missing values
        missing_before = df.isnull().sum()
        df = df.dropna()
        print(f"\n📉 Removed {initial_count - len(df):,} samples with missing values")
        
        # Remove outliers based on physical limits
        outlier_mask = np.zeros(len(df), dtype=bool)
        
        for target, limits in self.PHYSICAL_LIMITS.items():
            if target in df.columns:
                target_outliers = (df[target] < limits['min']) | (df[target] > limits['max'])
                n_outliers = target_outliers.sum()
                if n_outliers > 0:
                    print(f"   {target}: {n_outliers:,} outliers ({limits['min']}-{limits['max']})")
                    outlier_mask |= target_outliers.values
        
        df = df[~outlier_mask]
        print(f"\n📉 Removed {outlier_mask.sum():,} samples with out-of-range values")
        
        # Check physics constraint: PCE ≈ (Voc × Jsc × FF) / Pin
        if all(col in df.columns for col in ['PCE', 'Voc', 'Jsc', 'FF']):
            df['PCE_theoretical'] = (df['Voc'] * df['Jsc'] * df['FF']) / 100 * 100  # Pin = 100 mW/cm²
            df['PCE_violation'] = np.abs(df['PCE'] - df['PCE_theoretical'])
            
            # Remove samples with large physics violations (> 10% absolute)
            physics_violation_mask = df['PCE_violation'] > 10
            n_physics_violations = physics_violation_mask.sum()
            
            if n_physics_violations > 0:
                print(f"\n⚠️  {n_physics_violations:,} samples with physics constraint violations (> 10%)")
                df = df[~physics_violation_mask].drop(columns=['PCE_theoretical', 'PCE_violation'])
        
        # Compute statistics
        self.stats = {
            'initial_samples': initial_count,
            'final_samples': len(df),
            'removed_samples': initial_count - len(df),
            'removal_rate': (initial_count - len(df)) / initial_count * 100,
            'targets': list(df.columns)
        }
        
        self.df_clean = df.reset_index(drop=True)
        
        print(f"\n✅ Final dataset: {len(self.df_clean):,} samples")
        print(f"   Removal rate: {self.stats['removal_rate']:.1f}%")
        
        return self.df_clean, self.stats
    
    def analyze_correlations(self) -> Dict:
        """
        Analyze correlations between target variables.
        
        Returns:
            Dictionary with correlation matrices and statistics
        """
        print("\n" + "=" * 60)
        print("Target Correlation Analysis")
        print("=" * 60)
        
        targets = ['PCE', 'Voc', 'Jsc', 'FF']
        available_targets = [t for t in targets if t in self.df_clean.columns]
        
        # Pearson correlation
        corr_matrix = self.df_clean[available_targets].corr(method='pearson')
        
        print("\n📈 Pearson Correlation Matrix:")
        print(corr_matrix.round(3).to_string())
        
        # Spearman correlation (for non-linear relationships)
        corr_spearman = self.df_clean[available_targets].corr(method='spearman')
        
        # Compute partial correlations
        from scipy import stats
        
        partial_corr = {}
        for i, t1 in enumerate(available_targets):
            for t2 in available_targets[i+1:]:
                # Partial correlation controlling for other targets
                control = [t for t in available_targets if t not in [t1, t2]]
                
                if control:
                    # Compute residuals
                    from sklearn.linear_model import LinearRegression
                    
                    # Residualize t1
                    lr1 = LinearRegression()
                    lr1.fit(self.df_clean[control], self.df_clean[t1])
                    res1 = self.df_clean[t1] - lr1.predict(self.df_clean[control])
                    
                    # Residualize t2
                    lr2 = LinearRegression()
                    lr2.fit(self.df_clean[control], self.df_clean[t2])
                    res2 = self.df_clean[t2] - lr2.predict(self.df_clean[control])
                    
                    # Correlation of residuals
                    partial_corr[f"{t1}-{t2}"] = np.corrcoef(res1, res2)[0, 1]
        
        # Store results
        correlation_results = {
            'pearson': corr_matrix,
            'spearman': corr_spearman,
            'partial': partial_corr,
            'strong_correlations': []
        }
        
        # Identify strong correlations
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    correlation_results['strong_correlations'].append({
                        'pair': f"{corr_matrix.index[i]}-{corr_matrix.columns[j]}",
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        print(f"\n🔗 Strong correlations (|r| > 0.5):")
        for sc in correlation_results['strong_correlations']:
            print(f"   {sc['pair']}: r = {sc['correlation']:.3f}")
        
        self.correlation_results = correlation_results
        return correlation_results
    
    def create_visualizations(self, output_dir: Optional[Path] = None):
        """
        Create publication-quality visualizations.
        
        Args:
            output_dir: Directory to save figures
        """
        output_dir = output_dir or Path('reports/multi_objective/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        targets = ['PCE', 'Voc', 'Jsc', 'FF']
        available_targets = [t for t in targets if t in self.df_clean.columns]
        
        # 1. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = self.df_clean[available_targets].corr()
        
        sns.heatmap(
            corr, 
            annot=True, 
            fmt='.3f', 
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax
        )
        ax.set_title('Target Variable Correlations', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'target_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Saved: target_correlation_heatmap.png")
        
        # 2. Pairwise Scatter Plots
        fig = sns.pairplot(
            self.df_clean[available_targets],
            diag_kind='kde',
            plot_kws={'alpha': 0.5, 's': 10},
            diag_kws={'linewidth': 2}
        )
        fig.fig.suptitle('Target Variable Distributions and Relationships', y=1.02, fontsize=14, fontweight='bold')
        plt.savefig(output_dir / 'target_pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: target_pairplot.png")
        
        # 3. Distribution Plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, target in enumerate(available_targets):
            ax = axes[idx]
            data = self.df_clean[target]
            
            # Histogram with KDE
            sns.histplot(data, bins=50, kde=True, ax=ax, color=f'C{idx}')
            ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
            ax.axvline(data.median(), color='green', linestyle=':', label=f'Median: {data.median():.2f}')
            ax.set_xlabel(target, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.legend()
            
            # Add statistics
            stats_text = f'μ={data.mean():.2f}\nσ={data.std():.2f}\nn={len(data):,}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Target Variable Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'target_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: target_distributions.png")
        
        # 4. Physics Constraint Validation
        if all(t in self.df_clean.columns for t in targets):
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Compute theoretical PCE
            pce_theoretical = (self.df_clean['Voc'] * self.df_clean['Jsc'] * self.df_clean['FF']) / 100 * 100
            
            ax.scatter(self.df_clean['PCE'], pce_theoretical, alpha=0.3, s=10)
            ax.plot([0, 35], [0, 35], 'r--', label='Perfect Agreement')
            ax.set_xlabel('Measured PCE (%)', fontsize=11)
            ax.set_ylabel('Theoretical PCE (%)', fontsize=11)
            ax.set_title('Physics Constraint Validation: PCE = (Voc × Jsc × FF) / Pin', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Add R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(self.df_clean['PCE'], pce_theoretical)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(output_dir / 'physics_constraint_validation.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Saved: physics_constraint_validation.png")
        
        print(f"\n✅ All visualizations saved to {output_dir}")
    
    def save_cleaned_data(self, output_path: Optional[Path] = None):
        """Save cleaned multi-target data."""
        output_path = output_path or self.output_dir / 'multi_target_data.csv'
        
        self.df_clean.to_csv(output_path, index=False)
        print(f"\n💾 Saved cleaned data to: {output_path}")
        print(f"   Shape: {self.df_clean.shape}")
        
    def generate_report(self, output_path: Optional[Path] = None):
        """Generate comprehensive data preparation report."""
        output_path = output_path or Path('reports/multi_objective/target_correlation.md')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = f"""# Multi-Objective Target Correlation Analysis

**Generated**: 2026-03-12
**Research**: Physics-Informed Multi-Output Learning for Perovskite Solar Cells

---

## 📊 Data Summary

| Metric | Value |
|--------|-------|
| Initial samples | {self.stats['initial_samples']:,} |
| Final samples | {self.stats['final_samples']:,} |
| Removed samples | {self.stats['removed_samples']:,} |
| Removal rate | {self.stats['removal_rate']:.1f}% |

---

## 🎯 Target Statistics

"""
        
        # Add target statistics
        targets = ['PCE', 'Voc', 'Jsc', 'FF']
        available_targets = [t for t in targets if t in self.df_clean.columns]
        
        stats_df = self.df_clean[available_targets].describe()
        report += stats_df.to_markdown() + "\n\n"
        
        # Add correlation matrix
        report += """---

## 📈 Correlation Matrix

"""
        report += self.correlation_results['pearson'].round(3).to_markdown() + "\n\n"
        
        # Add strong correlations
        report += """---

## 🔗 Strong Correlations

"""
        for sc in self.correlation_results['strong_correlations']:
            report += f"- **{sc['pair']}**: r = {sc['correlation']:.3f}\n"
        
        # Add physical insights
        report += """

---

## 🔬 Physical Insights

### PCE-Voc-Jsc-FF Relationship

The theoretical relationship is:

$$PCE = \\frac{{Voc \\times Jsc \\times FF}}{{P_{{in}}}}$$

where $P_{{in}} = 100$ mW/cm² (standard AM1.5G illumination).

### Key Findings

1. **Voc-Jsc Trade-off**: There is often a trade-off between open-circuit voltage and short-circuit current, influenced by bandgap.
2. **FF-PCE Correlation**: Fill factor shows moderate correlation with PCE, indicating its importance for device optimization.
3. **Multi-objective Optimization**: The interdependencies suggest multi-objective optimization is necessary for optimal device design.

---

## 📊 Figures

- `target_correlation_heatmap.png` - Correlation heatmap
- `target_pairplot.png` - Pairwise distributions
- `target_distributions.png` - Individual distributions
- `physics_constraint_validation.png` - Physics constraint validation

---

**Next Steps**: Build multi-output models with physics-informed constraints.
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n📝 Saved report to: {output_path}")


def main():
    """Main execution for data preparation."""
    # Initialize
    prep = MultiObjectiveDataPreparation(
        input_path='/home/yhm/desktop/code/perovskite-eda-research/data/processed/perovskite_cleaned.csv',
        output_dir='/home/yhm/desktop/code/perovskite-eda-research/.worktrees/issue-9/multi_objective/data'
    )
    
    # Execute pipeline
    prep.load_data()
    prep.extract_targets()
    prep.clean_data()
    prep.analyze_correlations()
    prep.create_visualizations(
        output_dir=Path('/home/yhm/desktop/code/perovskite-eda-research/.worktrees/issue-9/reports/multi_objective/figures')
    )
    prep.save_cleaned_data()
    prep.generate_report(
        output_path=Path('/home/yhm/desktop/code/perovskite-eda-research/.worktrees/issue-9/reports/multi_objective/target_correlation.md')
    )
    
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()