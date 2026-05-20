"""Chemical-group holdout validation for verified discovery models.

This module emits a diagnostic validation split that keeps related molecules in
the same side of the split. It is intentionally not external validation: every
row still comes from the verified training table.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


FeatureFn = Callable[[pd.Series], Any]

PREDICTION_COLUMNS = [
    "record_id",
    "smiles",
    "chemical_group_key",
    "chemical_group_source",
    "y_true",
    "y_pred",
    "residual",
    "split",
]
SCHEMA_VERSION = "model-scaffold-validation-metrics-v1"
VALIDATION_SCOPE = "chemical_group_holdout_verified_rows"
SPLIT_STRATEGY = "rdkit_murcko_or_anonymous_graph_group_holdout"


@dataclass(frozen=True)
class ChemicalGroupAssignment:
    """Stable group assignment used for group-holdout validation."""

    key: str
    source: str


def evaluate_chemical_group_holdout(
    verified: pd.DataFrame,
    *,
    model: Any,
    feature_fn: FeatureFn,
    dataset_id: str,
    target_column: str,
    smiles_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate model performance on a chemical-group holdout split.

    The grouping prefers RDKit Murcko scaffolds. Molecules with no Murcko
    scaffold, such as acyclic additives, fall back to RDKit anonymous graph
    hashes. If RDKit is unavailable, a deterministic SMILES skeleton fallback is
    recorded explicitly in the metrics.
    """

    train_df = verified.dropna(subset=[smiles_column, target_column]).copy()
    train_df[target_column] = pd.to_numeric(train_df[target_column], errors="coerce")
    train_df = train_df.dropna(subset=[target_column]).reset_index(drop=True)
    assignments = [_chemical_group_assignment(value) for value in train_df[smiles_column]]
    train_df["chemical_group_key"] = [assignment.key for assignment in assignments]
    train_df["chemical_group_source"] = [assignment.source for assignment in assignments]
    unique_groups = sorted(train_df["chemical_group_key"].dropna().astype(str).unique().tolist())

    base_metrics = _base_metrics(
        dataset_id=dataset_id,
        target_column=target_column,
        smiles_column=smiles_column,
        test_size=test_size,
        random_state=random_state,
        train_df=train_df,
        unique_groups=unique_groups,
    )
    if len(train_df) < 2:
        return _skipped(base_metrics, "fewer than two trainable verified rows"), _empty_predictions()
    if len(unique_groups) < 2:
        return _skipped(base_metrics, "fewer than two chemical groups"), _empty_predictions()

    try:
        train_idx, validation_idx = next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state,
            ).split(train_df, groups=train_df["chemical_group_key"])
        )
    except ValueError as exc:
        return _skipped(base_metrics, f"group split failed: {exc}"), _empty_predictions()
    if len(train_idx) == 0 or len(validation_idx) == 0:
        return _skipped(base_metrics, "group split produced an empty side"), _empty_predictions()

    features = feature_fn(train_df[smiles_column])
    feature_frame, feature_columns = _as_feature_frame(features)
    y = train_df[target_column].to_numpy(dtype=float)
    X = feature_frame.to_numpy(dtype=float)

    fitted_model = _fresh_model(model)
    fitted_model.fit(X[train_idx], y[train_idx])
    y_pred = np.asarray(fitted_model.predict(X[validation_idx]), dtype=float)
    y_true = y[validation_idx]
    residual = y_pred - y_true
    validation_rows = train_df.iloc[validation_idx].copy()
    train_groups = sorted(train_df.iloc[train_idx]["chemical_group_key"].astype(str).unique().tolist())
    validation_groups = sorted(validation_rows["chemical_group_key"].astype(str).unique().tolist())
    overlap_groups = sorted(set(train_groups) & set(validation_groups))
    predictions = pd.DataFrame(
        {
            "record_id": validation_rows.get("record_id", pd.Series(validation_rows.index)).astype(str).to_list(),
            "smiles": validation_rows[smiles_column].astype(str).to_list(),
            "chemical_group_key": validation_rows["chemical_group_key"].astype(str).to_list(),
            "chemical_group_source": validation_rows["chemical_group_source"].astype(str).to_list(),
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": residual,
            "split": ["validation"] * len(validation_idx),
        },
        columns=PREDICTION_COLUMNS,
    )
    metrics = {
        **base_metrics,
        "status": "passed",
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "feature_columns": feature_columns,
        "model_class": fitted_model.__class__.__name__,
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "overlap_groups": overlap_groups,
        "group_overlap_count": int(len(overlap_groups)),
        "validation_r2": _safe_r2(y_true, y_pred),
        "validation_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "validation_mae": float(mean_absolute_error(y_true, y_pred)),
        "validation_pearson_r": _safe_pearson(y_true, y_pred),
    }
    return metrics, predictions


def _base_metrics(
    *,
    dataset_id: str,
    target_column: str,
    smiles_column: str,
    test_size: float,
    random_state: int,
    train_df: pd.DataFrame,
    unique_groups: list[str],
) -> dict[str, Any]:
    sources = train_df["chemical_group_source"].value_counts().sort_index().to_dict()
    rdkit_state = _rdkit_state()
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "status": "pending",
        "validation_scope": VALIDATION_SCOPE,
        "split_strategy": SPLIT_STRATEGY,
        "external_validation": False,
        "independent_external_validation": False,
        "random_holdout": False,
        "chemical_generalization_diagnostic": True,
        "chemical_generalization_claim": "not_established",
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": 0,
        "validation_rows": 0,
        "target_column": target_column,
        "smiles_column": smiles_column,
        "chemical_group_column": "chemical_group_key",
        "feature_columns": [],
        "model_class": None,
        "unique_groups": int(len(unique_groups)),
        "train_groups": [],
        "validation_groups": [],
        "overlap_groups": [],
        "group_overlap_count": 0,
        "group_method": {
            "rdkit_available": rdkit_state["available"],
            "rdkit_error": rdkit_state.get("error"),
            "assignment_source_counts": {str(key): int(value) for key, value in sources.items()},
            "uses_murcko_scaffold": bool(sources.get("rdkit_murcko")),
            "uses_anonymous_graph_fallback": bool(sources.get("rdkit_anonymous_graph")),
            "uses_smiles_skeleton_fallback": any(str(key).startswith("fallback_") for key in sources),
        },
        "validation_r2": None,
        "validation_rmse": None,
        "validation_mae": None,
        "validation_pearson_r": None,
    }


def _skipped(metrics: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = dict(metrics)
    payload["status"] = "skipped"
    payload["skip_reason"] = reason
    return payload


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _chemical_group_assignment(smiles: Any) -> ChemicalGroupAssignment:
    smiles_value = "" if pd.isna(smiles) else str(smiles).strip()
    if not smiles_value:
        return ChemicalGroupAssignment("missing_smiles", "missing_smiles")

    tools = _rdkit_tools()
    if tools is not None:
        chem = tools["Chem"]
        murcko = tools["MurckoScaffold"]
        mol_hash = tools["rdMolHash"]
        mol = chem.MolFromSmiles(smiles_value)
        if mol is not None:
            scaffold_mol = murcko.GetScaffoldForMol(mol)
            scaffold = chem.MolToSmiles(scaffold_mol, canonical=True)
            if scaffold:
                return ChemicalGroupAssignment(f"murcko:{scaffold}", "rdkit_murcko")
            anonymous_graph = mol_hash.MolHash(mol, mol_hash.HashFunction.AnonymousGraph)
            return ChemicalGroupAssignment(
                f"anonymous_graph:{anonymous_graph}",
                "rdkit_anonymous_graph",
            )
        return ChemicalGroupAssignment(
            f"fallback_smiles_skeleton:{_smiles_skeleton(smiles_value)}",
            "fallback_invalid_smiles_skeleton",
        )

    return ChemicalGroupAssignment(
        f"fallback_smiles_skeleton:{_smiles_skeleton(smiles_value)}",
        "fallback_smiles_skeleton",
    )


@lru_cache(maxsize=1)
def _rdkit_tools() -> dict[str, Any] | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolHash
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return None
    return {"Chem": Chem, "MurckoScaffold": MurckoScaffold, "rdMolHash": rdMolHash}


def _rdkit_state() -> dict[str, Any]:
    tools = _rdkit_tools()
    if tools is not None:
        return {"available": True, "error": None}
    return {"available": False, "error": "rdkit import unavailable"}


def _smiles_skeleton(smiles: str) -> str:
    skeleton = re.sub(r"Cl|Br|[A-Z][a-z]?|[cnops]", "A", smiles)
    skeleton = re.sub(r"\s+", "", skeleton)
    return skeleton or "empty"


def _as_feature_frame(features: Any) -> tuple[pd.DataFrame, list[str]]:
    if isinstance(features, pd.DataFrame):
        frame = features.fillna(0).copy()
        columns = [str(column) for column in frame.columns]
        frame.columns = columns
        return frame, columns
    if isinstance(features, pd.Series):
        name = str(features.name or "feature_0")
        frame = features.fillna(0).to_frame(name=name)
        return frame, [name]
    array = np.asarray(features)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    columns = [f"feature_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns), columns


def _fresh_model(model: Any) -> Any:
    try:
        return clone(model)
    except Exception:
        return deepcopy(model)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2:
        return None
    return float(r2_score(y_true, y_pred))


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return None
    return float(np.corrcoef(y_true, y_pred)[0, 1])
