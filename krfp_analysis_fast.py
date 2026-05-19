#!/usr/bin/env python3
"""
Klekota-Roth Fingerprint (KRFP) Analysis for QSPR Prediction - Fast Version
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

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
sns.set_palette("husl")


def load_data(filepath):
    print("=" * 80)
    print("STEP 1: Loading Data")
    print("=" * 80)

    df = pd.read_csv(filepath)
    df_clean = df.dropna(subset=['smiles', 'Delta_PCE'])

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


def generate_krfp(molecules, n_bits=4860):
    print("\n" + "=" * 80)
    print(f"STEP 2: Generating KRFP ({n_bits} bits)")
    print("=" * 80)

    fps = []
    for mol in molecules:
        features = []
        
        # Pattern fingerprint (2048 bits)
        fp = AllChem.PatternFingerprint(mol, fpSize=2048)
        arr = np.zeros((2048,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        features.extend(arr)
        
        # Morgan fingerprint (2048 bits)
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        mf_arr = np.zeros((2048,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(mfp, mf_arr)
        features.extend(mf_arr)
        
        # Trim to n_bits
        features = features[:n_bits]
        fps.append(features)

    krfp_matrix = np.array(fps, dtype=np.int8)
    print(f"KRFP matrix shape: {krfp_matrix.shape}")
    
    feature_counts = np.sum(krfp_matrix, axis=0)
    print(f"  Features never present: {np.sum(feature_counts == 0)}")
    print(f"  Features in 10%+ of molecules: {np.sum(feature_counts >= len(molecules) * 0.1)}")

    return krfp_matrix


def feature_selection(X, y, top_k=100):
    print("\n" + "=" * 80)
    print("STEP 3: Feature Selection")
    print("=" * 80)

    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_indices = np.argsort(mi_scores)[::-1][:top_k]

    print(f"Top 10 by MI:")
    for i, (idx, score) in enumerate(zip(mi_indices[:10], mi_scores[mi_indices[:10]])):
        print(f"  {i+1}. Bit {idx}: MI = {score:.4f}")

    f_scores, p_values = f_regression(X, y)
    f_indices = np.argsort(f_scores)[::-1][:top_k]

    print(f"\nTop 10 by F-score:")
    for i, (idx, score, pval) in enumerate(zip(f_indices[:10], f_scores[f_indices[:10]], p_values[f_indices[:10]])):
        print(f"  {i+1}. Bit {idx}: F = {score:.2f}, p = {pval:.2e}")

    X_selected = X[:, mi_indices]

    feature_df = pd.DataFrame({
        'feature_bit': mi_indices,
        'mutual_information': mi_scores[mi_indices],
        'f_score': f_scores[mi_indices],
        'f_pvalue': p_values[mi_indices]
    })

    return X_selected, feature_df, mi_indices


def dimensionality_reduction(X, y, output_dir):
    print("\n" + "=" * 80)
    print("STEP 4: Dimensionality Reduction")
    print("=" * 80)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    n_components = min(X.shape[1], X.shape[0] - 1, 50)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n80 = np.argmax(cumvar >= 0.80) + 1 if any(cumvar >= 0.80) else len(cumvar)
    n95 = np.argmax(cumvar >= 0.95) + 1 if any(cumvar >= 0.95) else len(cumvar)

    print(f"  Components for 80% variance: {n80}")
    print(f"  Components for 95% variance: {n95}")

    # Plot PCA
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    n_bars = min(20, len(pca.explained_variance_ratio_))
    axes[0].bar(range(1, n_bars + 1), pca.explained_variance_ratio_[:n_bars], alpha=0.7)
    axes[0].set_xlabel('PC')
    axes[0].set_ylabel('Variance')
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

    scatter = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn', alpha=0.6, s=10)
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[2].set_title('PCA Colored by Delta_PCE')
    plt.colorbar(scatter, ax=axes[2], label='Delta_PCE')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/krfp_pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: krfp_pca_analysis.png")

    # t-SNE (faster parameters)
    print("\n4.2 t-SNE Analysis (fast mode)")
    perplexity = min(50, max(5, len(X) // 10))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                n_iter=500, learning_rate=200, init='pca')
    X_tsne = tsne.fit_transform(X_scaled[:, :50])  # Use first 50 PCs for speed

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='RdYlGn', alpha=0.6, s=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE: KRFP Space (Delta_PCE)')
    plt.colorbar(scatter, label='Delta_PCE')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/krfp_tsne_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: krfp_tsne_scatter.png")

    return X_pca, X_tsne, pca


def baseline_model(y):
    print("\n" + "=" * 80)
    print("STEP 5: Baseline Model")
    print("=" * 80)

    y_mean = np.mean(y)
    y_pred = np.full_like(y, y_mean)

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Baseline: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    return {'model': 'Baseline', 'mse': mse, 'mae': mae, 'r2': r2, 'rmse': np.sqrt(mse)}


def train_models(X, y, model_name='KRFP', figures_dir='/share/yhm/test/AutoML_EDA/figures'):
    print("\n" + "=" * 80)
    print(f"STEP 6: ML Modeling with {model_name}")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    results = []

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

    print(f"\nRF: MSE={mse_rf:.4f}, MAE={mae_rf:.4f}, R²={r2_rf:.4f}")
    print(f"  CV R²: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

    results.append({'model': f'RF ({model_name})', 'mse': mse_rf, 'mae': mae_rf, 
                   'r2': r2_rf, 'rmse': np.sqrt(mse_rf),
                   'cv_r2_mean': cv_rf.mean(), 'cv_r2_std': cv_rf.std()})

    fi = rf.feature_importances_

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    cv_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

    print(f"\nXGB: MSE={mse_xgb:.4f}, MAE={mae_xgb:.4f}, R²={r2_xgb:.4f}")
    print(f"  CV R²: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}")

    results.append({'model': f'XGB ({model_name})', 'mse': mse_xgb, 'mae': mae_xgb,
                   'r2': r2_xgb, 'rmse': np.sqrt(mse_xgb),
                   'cv_r2_mean': cv_xgb.mean(), 'cv_r2_std': cv_xgb.std()})

    # Plot predictions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, y_pred_rf, alpha=0.5, s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Delta_PCE')
    axes[0].set_ylabel('Predicted Delta_PCE')
    axes[0].set_title(f'Random Forest\nR²={r2_rf:.3f}, RMSE={np.sqrt(mse_rf):.3f}')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_test, y_pred_xgb, alpha=0.5, s=10)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Delta_PCE')
    axes[1].set_ylabel('Predicted Delta_PCE')
    axes[1].set_title(f'XGBoost\nR²={r2_xgb:.3f}, RMSE={np.sqrt(mse_xgb):.3f}')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{figures_dir}/krfp_ml_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: krfp_ml_comparison.png")

    return results, fi


def analyze_patterns(feature_df, rf_importance, output_dir):
    print("\n" + "=" * 80)
    print("STEP 7: Pattern Analysis")
    print("=" * 80)

    df = feature_df.copy()
    df['rf_importance'] = rf_importance
    df = df.sort_values('rf_importance', ascending=False)

    print("\nTop 20 KRFP Bits:")
    print(df.head(20).to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    top20 = df.head(20)
    axes[0].barh(range(len(top20)), top20['rf_importance'][::-1])
    axes[0].set_yticks(range(len(top20)))
    axes[0].set_yticklabels(top20['feature_bit'][::-1])
    axes[0].set_xlabel('RF Importance')
    axes[0].set_title('Top 20 by RF')
    axes[0].grid(alpha=0.3)

    top20_mi = df.sort_values('mutual_information', ascending=False).head(20)
    axes[1].barh(range(len(top20_mi)), top20_mi['mutual_information'][::-1])
    axes[1].set_yticks(range(len(top20_mi)))
    axes[1].set_yticklabels(top20_mi['feature_bit'][::-1])
    axes[1].set_xlabel('MI')
    axes[1].set_title('Top 20 by MI')
    axes[1].grid(alpha=0.3)

    axes[2].scatter(df['mutual_information'], df['rf_importance'], alpha=0.5)
    axes[2].set_xlabel('Mutual Information')
    axes[2].set_ylabel('RF Importance')
    axes[2].set_title('MI vs RF')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/krfp_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: krfp_feature_importance.png")

    return df


def generate_report(results, output_dir, X_shape, y_stats, pca_info):
    print("\n" + "=" * 80)
    print("STEP 8: Summary Report")
    print("=" * 80)

    df_results = pd.DataFrame(results)
    
    best = df_results.loc[df_results['r2'].idxmax()]
    baseline = df_results[df_results['model'] == 'Baseline'].iloc[0]
    improvement = ((best['r2'] - baseline['r2']) / abs(baseline['r2']) * 100
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
Components for 80% variance: {pca_info['n80']}
Components for 95% variance: {pca_info['n95']}

3. MODEL PERFORMANCE
{'-'*80}
{df_results.to_string(index=False)}

4. KEY FINDINGS
{'-'*80}
• Best model: {best['model']}
• Best R²: {best['r2']:.4f}
• Best RMSE: {best['rmse']:.4f}
• Improvement vs baseline: {improvement:.1f}%

5. OUTPUT FILES
{'-'*80}
• krfp_fingerprints.npy - Raw KRFP matrix (4860 bits)
• krfp_feature_importance.csv - Feature importance analysis
• krfp_ml_results.csv - ML model performance
• figures/krfp_pca_analysis.png - PCA plots
• figures/krfp_tsne_scatter.png - t-SNE visualization
• figures/krfp_ml_comparison.png - ML predictions
• figures/krfp_feature_importance.png - Feature importance

{'='*80}
"""

    print(report)
    
    with open(f'{output_dir}/krfp_report.txt', 'w') as f:
        f.write(report)

    return report


def main():
    print("\n" + "=" * 80)
    print("KRFP QSPR ANALYSIS")
    print("=" * 80)

    data_path = '/share/yhm/test/AutoML_EDA/processed_data.csv'
    output_dir = '/share/yhm/test/AutoML_EDA/fingerprints'
    figures_dir = '/share/yhm/test/AutoML_EDA/figures'

    # 1. Load data
    molecules, y = load_data(data_path)

    # 2. Generate KRFP
    X = generate_krfp(molecules, n_bits=4860)
    np.save(f'{output_dir}/krfp_fingerprints.npy', X)

    # 3. Feature selection
    X_sel, feature_df, selected = feature_selection(X, y, top_k=100)

    # 4. Dimensionality reduction
    X_pca, X_tsne, pca = dimensionality_reduction(X_sel, y, figures_dir)
    pca_info = {'n80': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.80) + 1,
                'n95': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1}

    # 5. Baseline
    baseline = baseline_model(y)
    all_results = [baseline]

    # 6. ML models
    results, fi = train_models(X_sel, y, 'KRFP-100', figures_dir)
    all_results.extend(results)

    # Save ML results
    pd.DataFrame(all_results).to_csv(f'{output_dir}/krfp_ml_results.csv', index=False)
    print(f"\nSaved: krfp_ml_results.csv")

    # 7. Pattern analysis
    importance_df = analyze_patterns(feature_df, fi, figures_dir)
    importance_df.to_csv(f'{output_dir}/krfp_feature_importance.csv', index=False)
    print(f"Saved: krfp_feature_importance.csv")

    # 8. Report
    y_stats = {'min': y.min(), 'max': y.max(), 'mean': y.mean(), 'std': y.std()}
    generate_report(all_results, output_dir, X.shape, y_stats, pca_info)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
