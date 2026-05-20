"""
explore_features.py

Layer 2 Feature Engineering Exploration
=======================================
Generates and compares molecular representations for perovskite additive/modulator data:
  F21 — RDKit descriptors (basic + full)
  F22 — Fingerprints (ECFP4, ECFP6, MACCS, KRFP, AtomPair, TopologicalTorsion)

Outputs saved in the same folder as this script.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, rdBase
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler

# Silence RDKit parsing noise
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

# Ensure project root is on path so we can import features.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_generators import generate_all_features

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data_cache.csv"
SYNTHETIC_N = 500
TARGET_COL = "delta_pce"
SMILES_COL = "smiles"
RANDOM_STATE = 42
MAX_SAMPLES = 200  # cap real data to keep runtime reasonable

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_or_create_data() -> pd.DataFrame:
    """Load real data if available; otherwise build a small synthetic molecular dataset."""
    if DATA_PATH.exists():
        print(f"[Data] Loading real data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        # Clean: drop rows with missing SMILES or target
        df = df.dropna(subset=[SMILES_COL, TARGET_COL]).reset_index(drop=True)
        # Filter out SMILES that RDKit cannot parse (keep valid only for cleaner analysis)
        mask = df[SMILES_COL].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
        df = df[mask].reset_index(drop=True)
        if len(df) > MAX_SAMPLES:
            df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[Data] Valid molecules after filtering: {len(df)} (capped at {MAX_SAMPLES})")
        return df

    print("[Data] Real data not found — creating synthetic molecular dataset.")
    np.random.seed(RANDOM_STATE)
    # Simple organic fragments to build plausible SMILES
    fragments = [
        "C", "CC", "CCC", "CCCC", "CCCCC",
        "c1ccccc1", "c1ccc(C)cc1", "c1cc(C)ccc1O",
        "CCO", "CCCO", "CCCCO", "CCN", "CCCN",
        "COC", "COCC", "OCCO", "NCCN",
        "C=C", "C=CC", "C#C", "C=CC=C",
        "CC(=O)O", "CC(=O)N", "c1ccncc1", "C1CCCCC1",
        "CS", "CCS", "c1ccc(Cl)cc1", "c1ccc(F)cc1",
        "CC(C)C", "C(C)(C)C", "CC(=O)OC", "CN(C)C",
    ]
    smiles_list = []
    targets = []
    for _ in range(SYNTHETIC_N):
        # Concatenate 1–3 fragments
        n_frag = np.random.randint(1, 4)
        smi = "".join(np.random.choice(fragments, n_frag, replace=False))
        # Heuristic target influenced by length / complexity
        base = np.random.normal(0, 1)
        complexity = len(smi) * 0.05
        targets.append(base + complexity + np.random.normal(0, 0.5))
        smiles_list.append(smi)
    df = pd.DataFrame({SMILES_COL: smiles_list, TARGET_COL: targets})
    return df


# ---------------------------------------------------------------------------
# Feature analysis helpers
# ---------------------------------------------------------------------------

def analyze_sparsity(df: pd.DataFrame, name: str, report: dict):
    """Record basic shape / sparsity stats."""
    n_samples, n_features = df.shape
    # For fingerprints, most values are 0; for descriptors, NaNs are the main sparsity concern
    nan_frac = df.isna().mean().mean()
    zero_frac = (df == 0).mean().mean()
    report[name] = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "nan_fraction": float(nan_frac),
        "zero_fraction": float(zero_frac),
    }
    print(f"  → {name}: {n_samples} samples × {n_features} features | "
          f"NaN={nan_frac:.2%} | zeros={zero_frac:.2%}")


def variance_threshold_analysis(df: pd.DataFrame, name: str, thresholds=(0.0, 0.01, 0.05)) -> dict:
    """Show how many features survive at different variance thresholds."""
    # Impute NaNs with median for this analysis only
    df_imputed = df.fillna(df.median())
    results = {}
    print(f"\n[VarianceThreshold] {name}")
    for thr in thresholds:
        vt = VarianceThreshold(threshold=thr)
        try:
            vt.fit(df_imputed)
            n_kept = len(vt.get_support(indices=True))
        except ValueError:
            n_kept = df_imputed.shape[1]
        results[str(thr)] = int(n_kept)
        print(f"  threshold={thr:.2f} → {n_kept} / {df_imputed.shape[1]} features kept")
    return results


def top_univariate_features(df: pd.DataFrame, y: pd.Series, name: str, k: int = 20) -> pd.DataFrame:
    """Return top-k features by univariate f_regression score."""
    df_clean = df.dropna(axis=1, how="all").fillna(df.median())
    if df_clean.shape[1] == 0:
        return pd.DataFrame()
    # Scale descriptors; fingerprints are already binary/integer
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    selector = SelectKBest(score_func=f_regression, k=min(k, df_clean.shape[1]))
    selector.fit(X_scaled, y)
    scores = selector.scores_
    top_idx = np.argsort(scores)[::-1][:k]
    top_df = pd.DataFrame({
        "feature": df_clean.columns[top_idx],
        "score": scores[top_idx],
        "rank": range(1, len(top_idx) + 1),
    })
    print(f"\n[UnivariateSelection] {name} — top {k} features by f_regression:")
    print(top_df.head(10).to_string(index=False))
    return top_df


def descriptor_correlation_heatmap(df_desc: pd.DataFrame, y: pd.Series, out_path: Path, max_features: int = 30):
    """Save a correlation heatmap for the descriptors most correlated with the target."""
    df_clean = df_desc.dropna(axis=1, how="all").fillna(df_desc.median())
    if df_clean.shape[1] == 0:
        return
    # Compute correlation with target
    corr_with_y = df_clean.corrwith(y).abs().sort_values(ascending=False)
    selected = corr_with_y.head(max_features).index.tolist()
    df_sel = df_clean[selected]
    corr_mat = df_sel.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_mat, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3, cbar_kws={"shrink": 0.7})
    plt.title(f"Descriptor Correlation Heatmap (top {len(selected)} by |corr| with target)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved descriptor correlation heatmap → {out_path}")


def plot_feature_set_comparison(summary: dict, out_path: Path):
    """Bar plot comparing dimensionality across feature sets."""
    names = list(summary.keys())
    dims = [summary[n]["n_features"] for n in names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, dims, color=sns.color_palette("husl", len(names)))
    plt.ylabel("Number of features")
    plt.title("Feature Set Dimensionality Comparison (Layer 2)")
    plt.xticks(rotation=45, ha="right")
    for bar, d in zip(bars, dims):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(dims) * 0.01,
                 str(d), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved dimensionality comparison → {out_path}")


def fingerprint_density_plot(fingerprint_dfs: dict[str, pd.DataFrame], out_path: Path, sample_n: int = 5):
    """Plot bit-density (fraction of 1s) for a sample of fingerprint sets."""
    fig, axes = plt.subplots(1, min(sample_n, len(fingerprint_dfs)), figsize=(16, 4))
    if len(fingerprint_dfs) == 1:
        axes = [axes]
    for ax, (name, df) in zip(axes, list(fingerprint_dfs.items())[:sample_n]):
        densities = df.mean(axis=0).values
        ax.hist(densities, bins=50, color="steelblue", edgecolor="black")
        ax.set_title(name)
        ax.set_xlabel("Bit density (mean value)")
        ax.set_ylabel("Count")
    plt.suptitle("Fingerprint Bit-Density Distributions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved fingerprint density plot → {out_path}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Layer 2 Feature Engineering Exploration")
    print("=" * 70)

    # 1. Load data
    df = load_or_create_data()
    smiles = df[SMILES_COL]
    y = df[TARGET_COL]
    print(f"[Data] Working with {len(smiles)} molecules, target='{TARGET_COL}'")

    # 2. Generate all features
    print("\n[Step 1/5] Generating feature sets …")
    feature_sets = generate_all_features(smiles)

    # 3. Dimensionality & sparsity comparison
    print("\n[Step 2/5] Dimensionality & sparsity comparison …")
    summary = {}
    for name, fmat in feature_sets.items():
        analyze_sparsity(fmat, name, summary)

    # 4. Variance analysis
    print("\n[Step 3/5] Variance-threshold analysis …")
    variance_reports = {}
    for name, fmat in feature_sets.items():
        variance_reports[name] = variance_threshold_analysis(fmat, name)

    # 5. Univariate feature selection (top-k per set)
    print("\n[Step 4/5] Univariate feature selection (f_regression) …")
    selection_reports = {}
    for name, fmat in feature_sets.items():
        top_df = top_univariate_features(fmat, y, name, k=20)
        if not top_df.empty:
            selection_reports[name] = top_df.to_dict(orient="records")

    # 6. Correlation heatmap for descriptors
    print("\n[Step 5/5] Generating visualizations …")
    desc_sets = {k: v for k, v in feature_sets.items() if k.startswith("Descriptors_")}
    fp_sets = {k: v for k, v in feature_sets.items() if k.startswith("FP_")}

    if desc_sets:
        for name, fmat in desc_sets.items():
            heatmap_path = OUTPUT_DIR / f"correlation_heatmap_{name}.png"
            descriptor_correlation_heatmap(fmat, y, heatmap_path, max_features=30)

    # Dimensionality comparison plot
    dim_plot_path = OUTPUT_DIR / "feature_set_dimensionality.png"
    plot_feature_set_comparison(summary, dim_plot_path)

    # Fingerprint density plots
    if fp_sets:
        fp_density_path = OUTPUT_DIR / "fingerprint_densities.png"
        fingerprint_density_plot(fp_sets, fp_density_path, sample_n=6)

    # 7. Persist artefacts
    print("\n[Persist] Saving artefacts …")

    # Save summary JSON
    full_report = {
        "data_info": {
            "n_samples": len(df),
            "target_column": TARGET_COL,
            "smiles_column": SMILES_COL,
            "source": "data_cache.csv" if DATA_PATH.exists() else "synthetic",
        },
        "dimensionality_summary": summary,
        "variance_threshold_analysis": variance_reports,
    }
    report_path = OUTPUT_DIR / "feature_exploration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"  → {report_path}")

    # Save individual feature matrices as CSV (compressed via gz to save space for fingerprints)
    for name, fmat in feature_sets.items():
        csv_path = OUTPUT_DIR / f"{name}.csv.gz"
        fmat.to_csv(csv_path, compression="gzip", index=False)
        print(f"  → {csv_path}  ({fmat.shape[0]}×{fmat.shape[1]})")

    # Save top-selected features per set as CSV
    for name, records in selection_reports.items():
        sel_path = OUTPUT_DIR / f"top_features_{name}.csv"
        pd.DataFrame(records).to_csv(sel_path, index=False)
        print(f"  → {sel_path}")

    # 8. Console summary
    print("\n" + "=" * 70)
    print("Exploration complete. Summary:")
    print("=" * 70)
    print(f"{'Feature Set':<30} {'Samples':>8} {'Features':>10} {'NaN%':>8} {'Zero%':>8}")
    print("-" * 70)
    for name, info in summary.items():
        print(f"{name:<30} {info['n_samples']:>8} {info['n_features']:>10} "
              f"{info['nan_fraction']*100:>7.1f}% {info['zero_fraction']*100:>7.1f}%")
    print("=" * 70)
    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
