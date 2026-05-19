"""
explore_evaluation.py
=====================
Main exploration script for Layer 4 — Evaluation Strategies.

Demonstrates:
- E42: random split, scaffold split, temporal split
- E43: k-fold CV, repeated k-fold CV, nested CV
- E45: R², RMSE, MAE, uncertainty estimates (ensemble variance, bootstrap CI)

Compares how the choice of evaluation strategy affects reported model
performance on a synthetic molecular regression task.
"""

import json
import os
import sys
import time
import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

# ---------------------------------------------------------------------------
# Ensure imports from sibling modules work when run directly
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from splitters import (
    KFoldSplitter,
    NestedCVSplitter,
    RandomSplitter,
    RepeatedKFoldSplitter,
    ScaffoldKFoldSplitter,
    ScaffoldSplitter,
    TemporalSplitter,
)
from metrics import (
    EnsembleUncertainty,
    BootstrapUncertainty,
    aggregate_cv_results,
    mae,
    rmse,
    r2,
    print_metric_dict,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = SCRIPT_DIR
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Synthetic molecular data generation
# ---------------------------------------------------------------------------

SIMPLE_SMILES_FRAGMENTS = [
    "C", "CC", "CCC", "CCCC", "CCCCC", "c1ccccc1", "CC(C)C", "CCO", "CCN",
    "c1ccc(C)cc1", "C1CCCCC1", "CC(=O)O", "CC(C)(C)C", "c1ccccc1C",
    "CCOc1ccccc1", "c1ccc(OC)cc1", "CC(C)O", "CC(C)N", "CC(C)C(=O)O",
    "c1ccc(CC)cc1", "CCc1ccccc1", "CNC", "CC(C)NC", "c1ccc(CN)cc1",
    "CC(=O)N(C)C", "COC", "CN(C)C", "CC(C)C(=O)OC", "c1ccc(C(=O)O)cc1",
    "CC(=O)c1ccccc1", "c1ccc(C(=O)c2ccccc2)cc1", "CC(C)=O",
    "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(Cl)cc1", "c1ccc(Br)cc1",
    "c1ccc([N+](=O)[O-])cc1", "c1ccc(C(F)(F)F)cc1", "CCOC(=O)C",
    "c1ccc2ccccc2c1", "C1=CC=CC=C1", "CC#N", "C1CCOC1", "CCS",
    "c1ccc(S)cc1", "c1ccc(CS)cc1", "CC(C)(C)N", "CC(C)(C)O",
    "c1ccc(C(C)(C)C)cc1", "c1ccc(OC(C)C)cc1", "CC(C)Oc1ccccc1",
    "CCOc1ccc(C)cc1", "c1ccc(OCC)cc1", "CC(C)C(=O)N(C)C",
    "CCN(CC)CC", "c1ccc(N(C)C)cc1", "CC(C)N(C)C", "c1ccc(CN(C)C)cc1",
    "CC(=O)N(C)Cc1ccccc1", "c1ccc(CC(=O)N(C)C)cc1", "CC(C)CC(C)C",
    "CCCCCC", "CCCCCCC", "CCCCCCCC", "c1ccc(CCC)cc1", "CC(C)CCC(C)C",
    "c1ccc(C(C)C)cc1", "c1ccc(CC(C)C)cc1", "CC(C)Cc1ccccc1",
    "CCc1ccc(C)cc1", "Cc1ccc(C)cc1", "Cc1ccc(CC)cc1", "Cc1ccccc1C",
    "c1ccccc1CCc1ccccc1", "c1ccc(Cc2ccccc2)cc1", "CC(C)(c1ccccc1)c2ccccc2",
    "c1ccc(C(C)(C)c2ccccc2)cc1", "c1ccc(C(C)c2ccccc2)cc1",
    "CC(C)c1ccc(C(C)C)cc1", "CC(C)c1ccc(C(C)(C)C)cc1",
    "c1ccc(C(C)(C)c2ccc(C(C)(C)C)cc2)cc1", "c1ccc(C(C)c2ccc(C)cc2)cc1",
    "CC(C)c1ccc(CC(C)C)cc1", "CC(C)Cc1ccc(C(C)C)cc1",
    "c1ccc(CCCc2ccccc2)cc1", "c1ccc(CCc2ccc(C)cc2)cc1",
    "CC(C)Cc1ccc(CC(C)C)cc1", "c1ccc(C(C)CCc2ccccc2)cc1",
    "c1ccc(CC(C)(C)c2ccccc2)cc1", "c1ccc(C(C)(C)CCc2ccccc2)cc1",
]


def _mutate_smiles(smiles: str, rng: np.random.RandomState) -> str:
    """Apply a random structural mutation to a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    # simple strategy: add a random fragment
    frags = ["C", "O", "N", "S", "Cl", "F", "c1ccccc1", "C(C)(C)C", "OC"]
    frag = rng.choice(frags)
    new_smiles = smiles + frag
    new_mol = Chem.MolFromSmiles(new_smiles)
    if new_mol is not None:
        return Chem.MolToSmiles(new_mol)
    return smiles


def generate_synthetic_molecular_data(
    n_samples: int = 1000,
    n_features: int = 32,
    noise: float = 2.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic molecular regression data with SMILES strings and
    RDKit descriptors as features.  Target is a noisy linear combination
    of a few key descriptors plus noise.
    """
    rng = np.random.RandomState(random_state)

    # Generate diverse SMILES by mutating seed fragments
    smiles_pool = list(SIMPLE_SMILES_FRAGMENTS)
    while len(smiles_pool) < n_samples:
        base = rng.choice(SIMPLE_SMILES_FRAGMENTS)
        mutated = _mutate_smiles(base, rng)
        smiles_pool.append(mutated)
    smiles_list = rng.choice(smiles_pool, size=n_samples, replace=True).tolist()

    # Compute RDKit descriptors
    desc_names = [
        "MolWt",
        "MolLogP",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "TPSA",
        "NumAromaticRings",
        "NumSaturatedRings",
        "NumAliphaticRings",
        "NumValenceElectrons",
        "NumRadicalElectrons",
        "HeavyAtomCount",
        "NHOHCount",
        "NOCount",
        "RingCount",
    ]
    records = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        row = {"smiles": smi}
        for name in desc_names:
            try:
                val = getattr(Descriptors, name)(mol)
            except Exception:
                val = np.nan
            row[name] = val
        # Morgan fingerprint (count) as extra numeric features
        fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=n_features)
        arr = np.zeros(n_features, dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        for i, v in enumerate(arr):
            row[f"fp_{i}"] = v
        records.append(row)
        valid_smiles.append(smi)

    df = pd.DataFrame(records)
    df = df.reset_index(drop=True)

    # Build target from a known subset of descriptors + noise
    feature_cols = desc_names + [f"fp_{i}" for i in range(n_features)]
    X = df[feature_cols].fillna(0).values

    # True coefficients (sparse)
    true_coef = np.zeros(X.shape[1])
    key_indices = [0, 1, 2, 3, 4, 5]  # MolWt, LogP, HDonors, HAcceptors, RotBonds, TPSA
    true_coef[key_indices] = np.array([0.05, -1.2, 0.8, 0.6, -0.4, 0.02])
    y_true = X @ true_coef
    y = y_true + rng.normal(0, noise, size=len(y_true))

    df["target"] = y
    df["year"] = rng.randint(2015, 2026, size=len(df))  # for temporal split

    return df


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class SklearnModelWrapper:
    """Lightweight wrapper around scikit-learn regressors."""

    def __init__(self, model_cls, **kwargs):
        self.model_cls = model_cls
        self.kwargs = kwargs
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = self.model_cls(**self.kwargs)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    strategy: str
    splitter: str
    model: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0


def _get_feature_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
):
    X = df[feature_cols].fillna(0).values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    return X, y


def run_simple_split_evaluation(
    df: pd.DataFrame,
    splitter,
    model,
    feature_cols: List[str],
    target_col: str = "target",
    strategy_name: str = "simple_split",
    n_bootstrap: int = 1000,
) -> EvaluationResult:
    """Train / val / test split evaluation."""
    t0 = time.time()
    train_idx, val_idx, test_idx = splitter.split(df)

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy() if len(val_idx) > 0 else pd.DataFrame(columns=df.columns)
    test_df = df.iloc[test_idx].copy()

    X_train, y_train = _get_feature_target(train_df, feature_cols, target_col)
    X_val, y_val = _get_feature_target(val_df, feature_cols, target_col)
    X_test, y_test = _get_feature_target(test_df, feature_cols, target_col)

    model.fit(X_train, y_train)

    def _eval(X, y, prefix):
        preds = model.predict(X)
        return {
            f"{prefix}_r2": r2(y, preds),
            f"{prefix}_rmse": rmse(y, preds),
            f"{prefix}_mae": mae(y, preds),
        }

    metrics = {}
    if len(val_df) > 0:
        metrics.update(_eval(X_val, y_val, "val"))
    metrics.update(_eval(X_test, y_test, "test"))

    # Bootstrap CI on test set
    boot = BootstrapUncertainty(n_bootstrap=min(n_bootstrap, 200), random_state=RANDOM_STATE)
    test_preds = model.predict(X_test)
    _, std_r2, ci_r2 = boot.estimate(y_test, test_preds, r2)
    _, std_rmse, ci_rmse = boot.estimate(y_test, test_preds, rmse)
    _, std_mae, ci_mae = boot.estimate(y_test, test_preds, mae)

    uncertainty = {
        "test_r2": {"std": std_r2, "ci95": ci_r2},
        "test_rmse": {"std": std_rmse, "ci95": ci_rmse},
        "test_mae": {"std": std_mae, "ci95": ci_mae},
    }

    result = EvaluationResult(
        strategy=strategy_name,
        splitter=splitter.__class__.__name__,
        model=model.model_cls.__name__,
        metrics=metrics,
        uncertainty=uncertainty,
        runtime_seconds=time.time() - t0,
    )
    return result


def run_kfold_evaluation(
    df: pd.DataFrame,
    cv_splitter,
    model_factory,
    feature_cols: List[str],
    target_col: str = "target",
    strategy_name: str = "kfold",
    use_ensemble_uncertainty: bool = True,
) -> EvaluationResult:
    """K-fold or repeated k-fold CV evaluation."""
    t0 = time.time()
    X_all, y_all = _get_feature_target(df, feature_cols, target_col)

    fold_metrics = []
    fold_predictions = []  # for ensemble uncertainty
    fold_test_indices = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X_all, y_all)):
        model = model_factory()
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "r2": r2(y_test, preds),
                "rmse": rmse(y_test, preds),
                "mae": mae(y_test, preds),
            }
        )
        if use_ensemble_uncertainty:
            fold_predictions.append((test_idx, preds))
        fold_test_indices.append(test_idx)

    aggregated = aggregate_cv_results(fold_metrics)
    metrics = {k: v["mean"] for k, v in aggregated.items()}
    metrics["r2_std"] = aggregated.get("r2", {}).get("std", 0.0)
    metrics["rmse_std"] = aggregated.get("rmse", {}).get("std", 0.0)
    metrics["mae_std"] = aggregated.get("mae", {}).get("std", 0.0)

    uncertainty = {}
    if use_ensemble_uncertainty and fold_predictions:
        # Build per-sample ensemble predictions matrix
        n_samples = len(df)
        n_folds = len(fold_predictions)
        ensemble_matrix = np.full((n_folds, n_samples), np.nan)
        for i, (test_idx, preds) in enumerate(fold_predictions):
            ensemble_matrix[i, test_idx] = preds

        # Only samples that appear in >=2 folds
        valid_mask = np.sum(~np.isnan(ensemble_matrix), axis=0) >= 2
        if valid_mask.any():
            valid_preds = ensemble_matrix[:, valid_mask]
            mean_pred, std_pred, pred_interval = EnsembleUncertainty.from_predictions(
                valid_preds
            )
            y_valid = y_all[valid_mask]
            uncertainty["ensemble"] = {
                "n_samples": int(valid_mask.sum()),
                "mean_std": float(np.mean(std_pred)),
                "coverage_95": EnsembleUncertainty.coverage(y_valid, pred_interval),
                "interval_width": EnsembleUncertainty.interval_width(pred_interval),
                "nll": EnsembleUncertainty.negative_log_likelihood(
                    y_valid, mean_pred, std_pred
                ),
            }

    result = EvaluationResult(
        strategy=strategy_name,
        splitter=cv_splitter.__class__.__name__,
        model=model_factory().model_cls.__name__,
        metrics=metrics,
        fold_metrics=fold_metrics,
        uncertainty=uncertainty,
        runtime_seconds=time.time() - t0,
    )
    return result


def run_scaffold_kfold_evaluation(
    df: pd.DataFrame,
    cv_splitter: ScaffoldKFoldSplitter,
    model_factory,
    feature_cols: List[str],
    target_col: str = "target",
    strategy_name: str = "scaffold_kfold",
) -> EvaluationResult:
    """Scaffold-aware k-fold evaluation."""
    t0 = time.time()
    X_all, y_all = _get_feature_target(df, feature_cols, target_col)

    fold_metrics = []
    fold_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(df)):
        model = model_factory()
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "r2": r2(y_test, preds),
                "rmse": rmse(y_test, preds),
                "mae": mae(y_test, preds),
            }
        )
        fold_predictions.append((test_idx, preds))

    aggregated = aggregate_cv_results(fold_metrics)
    metrics = {k: v["mean"] for k, v in aggregated.items()}
    metrics["r2_std"] = aggregated.get("r2", {}).get("std", 0.0)
    metrics["rmse_std"] = aggregated.get("rmse", {}).get("std", 0.0)
    metrics["mae_std"] = aggregated.get("mae", {}).get("std", 0.0)

    uncertainty = {}
    n_samples = len(df)
    n_folds = len(fold_predictions)
    ensemble_matrix = np.full((n_folds, n_samples), np.nan)
    for i, (test_idx, preds) in enumerate(fold_predictions):
        ensemble_matrix[i, test_idx] = preds

    valid_mask = np.sum(~np.isnan(ensemble_matrix), axis=0) >= 2
    if valid_mask.any():
        valid_preds = ensemble_matrix[:, valid_mask]
        mean_pred, std_pred, pred_interval = EnsembleUncertainty.from_predictions(valid_preds)
        y_valid = y_all[valid_mask]
        uncertainty["ensemble"] = {
            "n_samples": int(valid_mask.sum()),
            "mean_std": float(np.mean(std_pred)),
            "coverage_95": EnsembleUncertainty.coverage(y_valid, pred_interval),
            "interval_width": EnsembleUncertainty.interval_width(pred_interval),
            "nll": EnsembleUncertainty.negative_log_likelihood(
                y_valid, mean_pred, std_pred
            ),
        }

    result = EvaluationResult(
        strategy=strategy_name,
        splitter=cv_splitter.__class__.__name__,
        model=model_factory().model_cls.__name__,
        metrics=metrics,
        fold_metrics=fold_metrics,
        uncertainty=uncertainty,
        runtime_seconds=time.time() - t0,
    )
    return result


def run_nested_cv_evaluation(
    df: pd.DataFrame,
    nested_splitter: NestedCVSplitter,
    model_factory,
    feature_cols: List[str],
    target_col: str = "target",
    strategy_name: str = "nested_cv",
) -> EvaluationResult:
    """
    Nested CV: outer loop for unbiased performance estimation.
    Inner loop is simulated here by a small hyper-parameter grid search
    (n_estimators for RandomForest).
    """
    t0 = time.time()
    X_all, y_all = _get_feature_target(df, feature_cols, target_col)

    # Simple inner-grid for demonstration
    inner_grid = [50, 100]

    outer_metrics = []
    for outer_train_idx, outer_test_idx in nested_splitter.outer_split(X_all, y_all):
        X_outer_train, y_outer_train = X_all[outer_train_idx], y_all[outer_train_idx]
        X_outer_test, y_outer_test = X_all[outer_test_idx], y_all[outer_test_idx]

        best_val_rmse = np.inf
        best_model = None

        for inner_train_idx, inner_val_idx in nested_splitter.inner_split(
            X_outer_train, y_outer_train
        ):
            X_inner_train, y_inner_train = (
                X_outer_train[inner_train_idx],
                y_outer_train[inner_train_idx],
            )
            X_inner_val, y_inner_val = (
                X_outer_train[inner_val_idx],
                y_outer_train[inner_val_idx],
            )

            for n_est in inner_grid:
                model = model_factory(n_estimators=n_est)
                model.fit(X_inner_train, y_inner_train)
                preds = model.predict(X_inner_val)
                val_rmse = rmse(y_inner_val, preds)
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_model = model

        # Evaluate best inner model on outer test fold
        test_preds = best_model.predict(X_outer_test)
        outer_metrics.append(
            {
                "r2": r2(y_outer_test, test_preds),
                "rmse": rmse(y_outer_test, test_preds),
                "mae": mae(y_outer_test, test_preds),
            }
        )

    aggregated = aggregate_cv_results(outer_metrics)
    metrics = {k: v["mean"] for k, v in aggregated.items()}
    metrics["r2_std"] = aggregated.get("r2", {}).get("std", 0.0)
    metrics["rmse_std"] = aggregated.get("rmse", {}).get("std", 0.0)
    metrics["mae_std"] = aggregated.get("mae", {}).get("std", 0.0)

    result = EvaluationResult(
        strategy=strategy_name,
        splitter=nested_splitter.__class__.__name__,
        model=best_model.model_cls.__name__ if best_model else "Unknown",
        metrics=metrics,
        fold_metrics=outer_metrics,
        uncertainty={},
        runtime_seconds=time.time() - t0,
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Layer 4 Evaluation Strategy Exploration")
    print("=" * 70)

    # 1. Data
    print("\n[1/4] Generating synthetic molecular regression data ...")
    df = generate_synthetic_molecular_data(
        n_samples=800, n_features=32, noise=2.0, random_state=RANDOM_STATE
    )
    feature_cols = [c for c in df.columns if c not in ("smiles", "target", "year")]
    print(f"       Samples: {len(df)} | Features: {len(feature_cols)} | SMILES column present")
    df.to_csv(OUTPUT_DIR / "synthetic_data.csv", index=False)
    print(f"       Saved: {OUTPUT_DIR / 'synthetic_data.csv'}")

    # 2. Define models
    def rf_factory(**extra):
        return SklearnModelWrapper(
            RandomForestRegressor,
            n_estimators=extra.get("n_estimators", 100),
            max_depth=extra.get("max_depth", 12),
            random_state=RANDOM_STATE,
            n_jobs=2,
        )

    results: List[EvaluationResult] = []

    # 3. Simple splits
    print("\n[2/4] Running simple train/val/test splits ...")

    for splitter, name in [
        (RandomSplitter(test_size=0.2, val_size=0.1, random_state=RANDOM_STATE), "random"),
        (ScaffoldSplitter(test_size=0.2, val_size=0.1, random_state=RANDOM_STATE), "scaffold"),
        (TemporalSplitter(test_size=0.2, val_size=0.1, time_col="year"), "temporal"),
    ]:
        res = run_simple_split_evaluation(
            df, splitter, rf_factory(), feature_cols, strategy_name=f"simple_split_{name}"
        )
        results.append(res)
        print(f"       {name:12s} | test R² = {res.metrics.get('test_r2', np.nan):.4f} | "
              f"RMSE = {res.metrics.get('test_rmse', np.nan):.4f}")

    # 4. K-fold CV strategies
    print("\n[3/4] Running cross-validation strategies ...")

    # Standard 5-fold
    kfold = KFoldSplitter(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    res = run_kfold_evaluation(
        df, kfold, rf_factory, feature_cols, strategy_name="5fold_cv"
    )
    results.append(res)
    print(f"       5-fold CV        | R² = {res.metrics['r2']:.4f} ± {res.metrics['r2_std']:.4f} | "
          f"RMSE = {res.metrics['rmse']:.4f} ± {res.metrics['rmse_std']:.4f}")

    # Repeated 5-fold (3 repeats)
    rep_kfold = RepeatedKFoldSplitter(
        n_splits=5, n_repeats=2, random_state=RANDOM_STATE
    )
    res = run_kfold_evaluation(
        df, rep_kfold, rf_factory, feature_cols, strategy_name="repeated_5fold_cv"
    )
    results.append(res)
    print(f"       Repeated 5-fold  | R² = {res.metrics['r2']:.4f} ± {res.metrics['r2_std']:.4f} | "
          f"RMSE = {res.metrics['rmse']:.4f} ± {res.metrics['rmse_std']:.4f}")

    # Scaffold 5-fold
    scaff_kfold = ScaffoldKFoldSplitter(
        n_splits=5, smiles_col="smiles", random_state=RANDOM_STATE
    )
    res = run_scaffold_kfold_evaluation(
        df, scaff_kfold, rf_factory, feature_cols, strategy_name="scaffold_5fold_cv"
    )
    results.append(res)
    print(f"       Scaffold 5-fold  | R² = {res.metrics['r2']:.4f} ± {res.metrics['r2_std']:.4f} | "
          f"RMSE = {res.metrics['rmse']:.4f} ± {res.metrics['rmse_std']:.4f}")

    # Nested CV
    nested = NestedCVSplitter(
        outer_n_splits=5, inner_n_splits=3, shuffle=True, random_state=RANDOM_STATE
    )
    res = run_nested_cv_evaluation(
        df, nested, rf_factory, feature_cols, strategy_name="nested_cv"
    )
    results.append(res)
    print(f"       Nested CV        | R² = {res.metrics['r2']:.4f} ± {res.metrics['r2_std']:.4f} | "
          f"RMSE = {res.metrics['rmse']:.4f} ± {res.metrics['rmse_std']:.4f}")

    # 5. Comparison summary
    print("\n[4/4] Comparison summary & saving results ...")

    summary_rows = []
    for res in results:
        row = {
            "strategy": res.strategy,
            "splitter": res.splitter,
            "model": res.model,
            "runtime_s": round(res.runtime_seconds, 2),
        }
        # Flatten metrics
        for k, v in res.metrics.items():
            row[f"metric_{k}"] = round(v, 4)
        # Flatten uncertainty
        for k, v in res.uncertainty.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    row[f"unc_{k}_{kk}"] = round(vv, 4) if isinstance(vv, float) else vv
            else:
                row[f"unc_{k}"] = v
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "evaluation_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"       Saved comparison CSV: {summary_path}")

    # JSON dump of full results
    json_path = OUTPUT_DIR / "evaluation_results.json"
    with open(json_path, "w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2, default=str)
    print(f"       Saved full JSON: {json_path}")

    # Pretty-print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    display_cols = [c for c in summary_df.columns if c.startswith("metric_")]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df[["strategy", "splitter", "model", "runtime_s"] + display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Random split typically yields the most optimistic R² because train and
   test molecules are drawn from the same distribution.

2. Scaffold split gives a more realistic (usually lower) estimate of
   generalisation because structurally similar molecules are kept together.

3. Temporal split mimics real-world deployment; performance drops if the
   data distribution shifts over time.

4. Repeated k-fold reduces variance in the CV estimate compared to a single
   k-fold run.

5. Nested CV gives an unbiased performance estimate but is computationally
   expensive; it is essential when hyper-parameter tuning is performed.

6. Ensemble uncertainty (std across folds) and bootstrap CIs quantify how
   much the reported metric may fluctuate with different data splits.
""")
    print("Done. All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
