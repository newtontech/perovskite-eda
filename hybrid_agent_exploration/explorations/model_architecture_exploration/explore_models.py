#!/usr/bin/env python3
"""
explore_models.py
Layer 3 (M31) model architecture exploration script.

Compares classical ML regressors on PSC additive prediction:
  - Random Forest
  - XGBoost
  - LightGBM
  - SVR
  - KNN

The script attempts to use real PSC data (RDKit descriptors from SMILES).
If insufficient valid SMILES are found, it falls back to synthetic regression
data that mimics the PSC additive prediction scenario (~5 354 samples,
chemical-like features, moderate noise).

Outputs (saved in the same folder):
  - model_comparison_results.csv
  - model_comparison_r2.png
  - per_fold_results.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_PATH = PROJECT_ROOT / "data_cache.csv"
OUTPUT_DIR = HERE

# ---------------------------------------------------------------------------
# RDKit descriptors (optional)
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import rdBase

    # Suppress verbose RDKit error messages during bulk SMILES parsing
    rdBase.DisableLog("rdApp.error")
    rdBase.DisableLog("rdApp.warning")

    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    RDKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
from model_factory import ModelFactory


def compute_rdkit_descriptors(smiles: str) -> Dict[str, float] | None:
    """Compute a basic set of RDKit descriptors for a SMILES string."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
        "NumRings": Descriptors.RingCount(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "MolMR": Descriptors.MolMR(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "BalabanJ": Descriptors.BalabanJ(mol),
    }


def load_real_data(min_valid: int = 100) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Load PSC data from data_cache.csv and compute RDKit descriptors.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable (delta_pce).
    source_note : str
        Description of the data source.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Expected columns per AGENTS.md
    if "delta_pce" not in df.columns and "Delta_PCE" in df.columns:
        df = df.rename(columns={"Delta_PCE": "delta_pce"})

    required = {"smiles", "delta_pce"}
    if not required.issubset(set(df.columns.str.lower())):
        raise ValueError(f"CSV missing required columns. Have: {list(df.columns)}")

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Filter valid SMILES and compute descriptors
    valid_rows: List[Dict[str, float]] = []
    valid_targets: List[float] = []

    for _, row in df.iterrows():
        smi = str(row["smiles"])
        desc = compute_rdkit_descriptors(smi)
        if desc is not None:
            valid_rows.append(desc)
            valid_targets.append(row["delta_pce"])

    if len(valid_rows) < min_valid:
        raise ValueError(
            f"Only {len(valid_rows)} valid SMILES found (min {min_valid})."
        )

    # Cap at 3 000 samples to keep CV runtime reasonable
    max_samples = 3000
    if len(valid_rows) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(valid_rows), size=max_samples, replace=False)
        valid_rows = [valid_rows[i] for i in idx]
        valid_targets = [valid_targets[i] for i in idx]

    X = pd.DataFrame(valid_rows)
    y = pd.Series(valid_targets, name="delta_pce")
    return X, y, f"Real PSC data (n={len(y)}, descriptors from RDKit)"



def generate_synthetic_data(
    n_samples: int = 5354, n_features: int = 10, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Generate synthetic regression data that mimics a PSC additive prediction scenario.

    Feature names are chosen to resemble molecular / process descriptors.
    """
    rng = np.random.RandomState(random_state)

    feature_names = [
        "molecular_weight",
        "tpsa",
        "log_p",
        "h_bond_donors",
        "h_bond_acceptors",
        "rotatable_bonds",
        "additive_concentration",
        "annealing_temp",
        "initial_pce",
        "ionic_radius_ratio",
    ][:n_features]

    X = rng.randn(n_samples, n_features)
    # Introduce some correlation structure
    X[:, 2] = 0.5 * X[:, 0] + 0.5 * rng.randn(n_samples)  # LogP ~ MolWt
    X[:, 4] = 0.3 * X[:, 3] + 0.7 * rng.randn(n_samples)  # HBA ~ HBD

    # Non-linear target: Delta_PCE driven by a subset of features + interaction
    true_coef = np.array([0.3, -0.2, 0.5, 0.4, 0.1, -0.15, 0.6, 0.25, -0.35, 0.2])
    interaction = 0.1 * X[:, 6] * X[:, 8]  # concentration × initial_pce interaction
    y_true = X @ true_coef[:n_features] + interaction
    noise = rng.normal(loc=0.0, scale=0.8, size=n_samples)
    y = y_true + noise

    # Clip to a realistic PCE delta range (-5 % to +8 %)
    y = np.clip(y, -5.0, 8.0)

    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y, name="delta_pce")
    return X_df, y_s, f"Synthetic PSC-like data (n={n_samples}, {n_features} features)"


def cross_validate(
    model_wrapper,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, List[float]]:
    """
    Perform n_splits-fold cross-validation.

    Returns per-fold metrics: r2, rmse, mae.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: Dict[str, List[float]] = {"r2": [], "rmse": [], "mae": []}

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model_wrapper.fit(X_train_s, y_train)
        y_pred = model_wrapper.predict(X_val_s)

        results["r2"].append(r2_score(y_val, y_pred))
        results["rmse"].append(root_mean_squared_error(y_val, y_pred))
        results["mae"].append(mean_absolute_error(y_val, y_pred))

    return results


def run_exploration() -> None:
    print("=" * 70)
    print("Layer 3 Model Architecture Exploration (M31)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load or synthesise data
    # ------------------------------------------------------------------
    try:
        X, y, data_note = load_real_data(min_valid=100)
    except Exception as exc:
        print(f"[INFO] Falling back to synthetic data. Reason: {exc}")
        X, y, data_note = generate_synthetic_data()

    print(f"\nData source: {data_note}", flush=True)
    print(f"Feature matrix shape: {X.shape}", flush=True)
    print(f"Target stats: mean={y.mean():.3f}, std={y.std():.3f}", flush=True)

    # ------------------------------------------------------------------
    # 2. Initialise model factory
    # ------------------------------------------------------------------
    factory = ModelFactory()
    model_names = factory.list_models()
    print(f"\nModels to evaluate: {model_names}", flush=True)

    # ------------------------------------------------------------------
    # 3. Cross-validate each model
    # ------------------------------------------------------------------
    summary_rows: List[Dict[str, float | str]] = []
    per_fold_records: List[Dict[str, float | str]] = []

    for mname in model_names:
        print(f"\n  ▶ Evaluating {mname} ...", end=" ", flush=True)
        model = factory.create(mname)
        cv_results = cross_validate(model, X.values, y.values, n_splits=5)

        # Aggregate
        row = {
            "model": mname,
            "cv_r2_mean": np.mean(cv_results["r2"]),
            "cv_r2_std": np.std(cv_results["r2"]),
            "cv_rmse_mean": np.mean(cv_results["rmse"]),
            "cv_rmse_std": np.std(cv_results["rmse"]),
            "cv_mae_mean": np.mean(cv_results["mae"]),
            "cv_mae_std": np.std(cv_results["mae"]),
        }
        summary_rows.append(row)

        # Per-fold records
        for fold_idx, (r2, rmse, mae) in enumerate(
            zip(cv_results["r2"], cv_results["rmse"], cv_results["mae"])
        ):
            per_fold_records.append(
                {
                    "model": mname,
                    "fold": fold_idx + 1,
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                }
            )
        print("done.", flush=True)

    # ------------------------------------------------------------------
    # 4. Summary table
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("cv_r2_mean", ascending=False).reset_index(
        drop=True
    )

    print("\n" + "=" * 70)
    print("Model Comparison Summary (5-Fold CV)")
    print("=" * 70)
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # 5. Save artefacts
    # ------------------------------------------------------------------
    summary_csv = OUTPUT_DIR / "model_comparison_results.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[INFO] Saved summary → {summary_csv}")

    per_fold_csv = OUTPUT_DIR / "per_fold_results.csv"
    pd.DataFrame(per_fold_records).to_csv(per_fold_csv, index=False)
    print(f"[INFO] Saved per-fold results → {per_fold_csv}")

    # ------------------------------------------------------------------
    # 6. Visualisation
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))

        x_pos = np.arange(len(summary_df))
        ax.bar(
            x_pos,
            summary_df["cv_r2_mean"],
            yerr=summary_df["cv_r2_std"],
            capsize=4,
            color="steelblue",
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary_df["model"], rotation=30, ha="right")
        ax.set_ylabel("CV R²")
        ax.set_title("Layer 3 Model Comparison — 5-Fold CV R²")
        ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylim(
            min(0, summary_df["cv_r2_mean"].min() - 0.05),
            summary_df["cv_r2_mean"].max() + 0.1,
        )

        plt.tight_layout()
        plot_path = OUTPUT_DIR / "model_comparison_r2.png"
        fig.savefig(plot_path, dpi=300)
        print(f"[INFO] Saved plot → {plot_path}")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Could not generate plot: {exc}")

    print("\n" + "=" * 70)
    print("Exploration complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_exploration()
