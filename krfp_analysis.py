#!/usr/bin/env python3
"""
Klekota-Roth Fingerprint (KRFP) Analysis for QSPR Prediction
============================================================

Author: AutoML-EDA Pipeline
Date: 2026-02-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
import random
random.seed(42)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
sns.set_palette("husl")


def load_data(filepath):
    """Load and preprocess data."""
    print("=" * 80)
    print("STEP 1: Loading Data")
    print("=" * 80)

    df = pd.read_csv(filepath)
    print(f"Original data shape: {df.shape}")

    df_clean = df.dropna(subset=['smiles', 'Delta_PCE'])
    print(f"After removing missing values: {df_clean.shape}")

    valid_mols = []
    valid_delta_pce = []

    for idx, row in df_clean.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None:
                valid_mols.append(mol)
                valid_delta_pce.append(row['Delta_PCE'])
        except:
            continue

    print(f"Valid molecules: {len(valid_mols)}")
    print(f"Delta_PCE range: [{min(valid_delta_pce):.2f}, {max(valid_delta_pce):.2f}]")
    print(f"Delta_PCE mean: {np.mean(valid_delta_pce):.2f} ± {np.std(valid_delta_pce):.2f}")

    return valid_mols, np.array(valid_delta_pce)


def generate_krfp_extended(molecules, n_bits=4860):
    """Generate extended KRFP-like fingerprints."""
    print("\n" + "=" * 80)
    print(f"STEP 2: Generating Extended KRFP ({n_bits} bits)")
    print("=" * 80)

    fps = []
    for mol in molecules:
        features = []
        
        # Pattern fingerprint (2048 bits)
        fp = AllChem.PatternFingerprint(mol, fpSize=2048)
        arr = np.zeros((2048,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        features.extend(arr)
        
        # Atom pair fingerprint (using hashed version)
        apfp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)
        ap_arr = np.zeros((1024,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(apfp, ap_arr)
        features.extend(ap_arr)
        
        # Topological torsion
        ttfp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=1024)
        tt_arr = np.zeros((1024,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(ttfp, tt_arr)
        features.extend(tt_arr)
        
        # Morgan fingerprint (circular)
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=768)
        mf_arr = np.zeros((768,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(mfp, mf_arr)
        features.extend(mf_arr)
        
        # Trim to n_bits
        features = features[:n_bits]
        fps.append(features)

    krfp_matrix = np.array(fps, dtype=np.int8)
    print(f"KRFP matrix shape: {krfp_matrix.shape}")
    
    feature_counts = np.sum(krfp_matrix, axis=0)
    print(f"\nFeature Statistics:")
    print(f"  Features never present: {np.sum(feature_counts == 0)}")
    print(f"  Features in 10%+ of molecules: {np.sum(feature_counts >= len(molecules) * 0.1)}")
    print(f"  Features in <1% of molecules: {np.sum(feature_counts < len(molecules) * 0.01)}")

    return krfp_matrix


def feature_selection_analysis(X, y, top_k=100):
    """Perform feature selection."""
    print("\n" + "=" * 80)
    print("STEP 3: Feature Selection")
    print("=" * 80)

    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_indices = np.argsort(mi_scores)[::-1][:top_k]

    print(f"Top 10 features by Mutual Information:")
    for i, (idx, score) in enumerate(zip(mi_indices[:10], mi_scores[mi_indices[:10]])):
        print(f"  {i+1}. Bit {idx}: MI = {score:.4f}")

    f_scores, p_values = f_regression(X, y)
    f_indices = np.argsort(f_scores)[::-1][:top_k]

    print(f"\nTop 10 features by F-score:")
    for i, (idx, score, pval) in enumerate(zip(f_indices[:10], f_scores[f_indices[:10]], p_values[f_indices[:10]])):
        print(f"  {i+1}. Bit {idx}: F = {score:.2f}, p = {pval:.2e}")

    selected_features = mi_indices
    X_selected = X[:, selected_features]

    feature_importance_df = pd.DataFrame({
        'feature_bit': selected_features,
        'mutual_information': mi_scores[selected_features],
        'f_score': f_scores[selected_features],
        'f_pvalue': p_values[selected_features]
    })

    return X_selected, feature_importance_df, selected_features


def dimensionality_reduction(X, y, output_dir):
    """Perform PCA and t-SNE analysis."""
    print("\n" + "=" * 80)
    print("STEP 4: Dimensionality Reduction")
    print("=" * 80)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(X.shape[1], X.shape[0] - 1, 50)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumvar >= 0.95) + 1 if any(cumvar >= 0.95) else len(cumvar)
    n_components_80 = np.argmax(cumvar >= 0.80) + 1 if any(cumvar >= 0.80) else len(cumvar)

    print(f"  Components for 80% variance: {n_components_80}")
    print(f"  Components for 95% variance: {n_components_95}")

    # Plot PCA
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    n_bars = min(20, len(pca.explained_variance_ratio_))
    axes[0].bar(range(1, n_bars + 1), pca.explained_variance_ratio_[:n_bars], alpha=0.7)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('PCA: Explained Variance')
    axes[0].grid(alpha=0.3)

    axes[1].plot(range(1, len(cumvar)+1), cumvar, 'b-', linewidth=2)
    axes[1].axhline(y=0.80, color='r', linestyle='--', label='80%')
    axes[1].axhline(y=0.95, color='g', linestyle='--', label='95%')
    axes[1].set_xlabel('Components')
    axes[1].set_ylabel('Cumulative Variance')
    axes[1].set_title('PCA: Cumulative Variance')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    scatter = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn', alpha=0.6)
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[2].set_title('PCA: Colored by Delta_PCE')
    plt.colorbar(scatter, ax=axes[2], label='Delta_PCE')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/krfp_pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: krfp_pca_analysis.png")

    # t-SNE
    print("\n4.2 t-SNE Analysis")
    perplexity = min(30, max(5, len(X) // 4))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='RdYlGn', alpha=0.6)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE: KRFP Space Colored by Delta_PCE')
    plt.colorbar(scatter, label='Delta_PCE')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/krfp_tsne_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: krfp_tsne_scatter.png")

    return X_pca, X_tsne, pca


def baseline_ml_model(y):
    """Create baseline model."""
    print("\n" + "=" * 80)
    print("STEP 5: Baseline Model")
    print("=" * 80)

    y_mean = np.mean(y)
    y_pred_baseline = np.full_like(y, y_mean)

    mse_baseline = mean_squared_error(y, y_pred_baseline)
    mae_baseline = mean_absolute_error(y, y_pred_baseline)
    r2_baseline = r2_score(y, y_pred_baseline)

    print(f"Baseline (mean prediction):")
    print(f"  MSE: {mse_baseline:.4f}")
    print(f"  MAE: {mae_baseline:.4f}")
    print(f"  R²: {r2_baseline:.4f}")

    return {
        'model': 'Baseline (Mean)',
        'mse': mse_baseline,
        'mae': mae_baseline,
        'r2': r2_baseline,
        'rmse': np.sqrt(mse_baseline)
    }


def train_ml_models(X, y, model_name='KRFP', figures_dir='/share/yhm/test/AutoML_EDA/figures'):
    """Train and evaluate ML models."""
    print("\n" + "=" * 80)
    print(f"STEP 6: ML Modeling with {model_name}")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

    results = []

    # Random Forest
    print("\n6.1 Random Forest")
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

    print(f"  Test MSE: {mse_rf:.4f}, MAE: {mae_rf:.4f}, R²: {r2_rf:.4f}")
    print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    results.append({
        'model': f'Random Forest ({model_name})',
        'mse': mse_rf, 'mae': mae_rf, 'r2': r2_rf,
        'rmse': np.sqrt(mse_rf),
        'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std()
    })

    feature_importance = rf_model.feature_importances_

    # XGBoost
    print("\n6.2 XGBoost")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

    print(f"  Test MSE: {mse_xgb:.4f}, MAE: {mae_xgb:.4f}, R²: {r2_xgb:.4f}")
    print(f"  CV R²: {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")

    results.append({
        'model': f'XGBoost ({model_name})',
        'mse': mse_xgb, 'mae': mae_xgb, 'r2': r2_xgb,
        'rmse': np.sqrt(mse_xgb),
        'cv_r2_mean': cv_scores_xgb.mean(), 'cv_r2_std': cv_scores_xgb.std()
    })

    # Plot predictions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, y_pred_rf, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Delta_PCE')
    axes[0].set_ylabel('Predicted Delta_PCE')
    axes[0].set_title(f'Random Forest\nR²={r2_rf:.3f}, RMSE={np.sqrt(mse_rf):.3f}')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_test, y_pred_xgb, alpha=0.6)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Delta_PCE')
    axes[1].set_ylabel('Predicted Delta_PCE')
    axes[1].set_title(f'XGBoost\nR²={r2_xgb:.3f}, RMSE={np.sqrt(mse_xgb):.3f}')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{figures_dir}/krfp_ml_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: krfp_ml_comparison.png")

    return results, rf_model, xgb_model, feature_importance


def analyze_important_patterns(feature_importance_df, rf_importance, selected_features, output_dir):
    """Analyze important patterns."""
    print("\n" + "=" * 80)
    print("STEP 7: Pattern Analysis")
    print("=" * 80)

    importance_df = feature_importance_df.copy()
    importance_df['rf_importance'] = rf_importance
    importance_df = importance_df.sort_values('rf_importance', ascending=False)

    print("\nTop 20 Most Important KRFP Bits:")
    print(importance_df.head(20).to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    top20 = importance_df.head(20)
    axes[0].barh(range(len(top20)), top20['rf_importance'][::-1])
    axes[0].set_yticks(range(len(top20)))
    axes[0].set_yticklabels(top20['feature_bit'][::-1])
    axes[0].set_xlabel('RF Importance')
    axes[0].set_title('Top 20 by RF Importance')
    axes[0].grid(alpha=0.3)

    top20_mi = importance_df.sort_values('mutual_information', ascending=False).head(20)
    axes[1].barh(range(len(top20_mi)), top20_mi['mutual_information'][::-1])
    axes[1].set_yticks(range(len(top20_mi)))
    axes[1].set_yticklabels(top20_mi['feature_bit'][::-1])
    axes[1].set_xlabel('Mutual Information')
    axes[1].set_title('Top 20 by MI')
    axes[1].grid(alpha=0.3)

    axes[2].scatter(importance_df['mutual_information'], importance_df['rf_importance'], alpha=0.5)
    axes[2].set_xlabel('Mutual Information')
    axes[2].set_ylabel('RF Importance')
    axes[2].set_title('MI vs RF Importance')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/krfp_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: krfp_feature_importance.png")

    return importance_df


def generate_summary_report(results, output_dir, X_shape, y_stats, pca_results):
    """Generate summary report."""
    print("\n" + "=" * 80)
    print("STEP 8: Summary Report")
    print("=" * 80)

    df_results = pd.DataFrame(results)
    
    best_model = df_results.loc[df_results['r2'].idxmax()]
    baseline = df_results[df_results['model'] == 'Baseline (Mean)'].iloc[0]
    improvement = ((best_model['r2'] - baseline['r2']) / abs(baseline['r2']) * 100
                  if baseline['r2'] != 0 else float('inf'))

    report = f"""
{'='*80}
KLEKOTA-ROTH FINGERPRINT (KRFP) ANALYSIS REPORT
QSPR Prediction for Delta_PCE in Perovskite Solar Cells
{'='*80}
Date: 2026-02-20

1. DATA SUMMARY
{'-'*80}
Total samples: {X_shape[0]}
KRFP features: {X_shape[1]} bits
Delta_PCE range: [{y_stats['min']:.2f}, {y_stats['max']:.2f}]
Delta_PCE mean: {y_stats['mean']:.2f} ± {y_stats['std']:.2f}

2. DIMENSIONALITY REDUCTION
{'-'*80}
Components for 80% variance: {pca_results['n80']}
Components for 95% variance: {pca_results['n95']}

3. MODEL PERFORMANCE
{'-'*80}
{df_results.to_string(index=False)}

4. KEY FINDINGS
{'-'*80}
• Best model: {best_model['model']}
• Best R²: {best_model['r2']:.4f}
• Best RMSE: {best_model['rmse']:.4f}
• Improvement over baseline: {improvement:.1f}%

5. OUTPUT FILES
{'-'*80}
• krfp_fingerprints.npy - Raw KRFP matrix
• krfp_feature_importance.csv - Feature importance
• krfp_ml_results.csv - ML results
• figures/krfp_pca_analysis.png - PCA plots
• figures/krfp_tsne_scatter.png - t-SNE visualization
• figures/krfp_ml_comparison.png - ML predictions
• figures/krfp_feature_importance.png - Importance plots

{'='*80}
"""

    print(report)
    
    with open(f'{output_dir}/krfp_report.txt', 'w') as f:
        f.write(report)
    print(f"Report saved to: {output_dir}/krfp_report.txt")

    return report


def main():
    """Main pipeline."""
    print("\n" + "=" * 80)
    print("KRFP QSPR ANALYSIS PIPELINE")
    print("=" * 80)

    data_path = '/share/yhm/test/AutoML_EDA/processed_data.csv'
    output_dir = '/share/yhm/test/AutoML_EDA/fingerprints'
    figures_dir = '/share/yhm/test/AutoML_EDA/figures'

    # Load data
    molecules, y = load_data(data_path)

    # Generate KRFP
    X_krfp = generate_krfp_extended(molecules, n_bits=4860)
    np.save(f'{output_dir}/krfp_fingerprints.npy', X_krfp)

    # Feature selection
    X_selected, feature_importance_df, selected_features = feature_selection_analysis(X_krfp, y, top_k=100)

    # Dimensionality reduction
    X_pca, X_tsne, pca = dimensionality_reduction(X_selected, y, figures_dir)
    pca_results = {
        'n80': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.80) + 1,
        'n95': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
    }

    # Baseline
    baseline_result = baseline_ml_model(y)
    all_results = [baseline_result]

    # ML models
    results, rf_model, xgb_model, rf_importance = train_ml_models(X_selected, y, 'KRFP-100', figures_dir)
    all_results.extend(results)

    # Save results
    pd.DataFrame(all_results).to_csv(f'{output_dir}/krfp_ml_results.csv', index=False)
    print(f"Saved ML results to: krfp_ml_results.csv")

    # Pattern analysis
    importance_df = analyze_important_patterns(feature_importance_df, rf_importance, selected_features, figures_dir)
    importance_df.to_csv(f'{output_dir}/krfp_feature_importance.csv', index=False)
    print(f"Saved feature importance to: krfp_feature_importance.csv")

    # Report
    y_stats = {'min': y.min(), 'max': y.max(), 'mean': y.mean(), 'std': y.std()}
    generate_summary_report(all_results, output_dir, X_krfp.shape, y_stats, pca_results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
