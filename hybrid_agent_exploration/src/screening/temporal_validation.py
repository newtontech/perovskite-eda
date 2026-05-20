"""Temporal holdout validation for verified discovery models.

This diagnostic trains on earlier verified literature rows and validates on
the newest verified rows. It is chronological stress testing, not independent
external validation or experimental feedback.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FeatureFn = Callable[[pd.Series], Any]

PREDICTION_COLUMNS = ["record_id", "smiles", "year", "y_true", "y_pred", "residual", "split"]
SCHEMA_VERSION = "model-temporal-validation-metrics-v1"
VALIDATION_SCOPE = "temporal_holdout_verified_rows"
SPLIT_STRATEGY = "latest_reference_year_holdout"


@dataclass(frozen=True)
class TemporalSplit:
    """Resolved chronological split."""

    train_mask: pd.Series
    validation_mask: pd.Series
    train_years: list[int]
    validation_years: list[int]


def evaluate_temporal_holdout(
    verified: pd.DataFrame,
    *,
    model: Any,
    feature_fn: FeatureFn,
    dataset_id: str,
    target_column: str,
    smiles_column: str,
    year_column: str = "year",
    validation_fraction: float = 0.2,
    min_validation_rows: int = 2,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate chronological generalization on the newest verified rows."""

    train_df = verified.dropna(subset=[smiles_column, target_column]).copy()
    train_df[target_column] = pd.to_numeric(train_df[target_column], errors="coerce")
    if year_column not in train_df.columns:
        metrics = _base_metrics(dataset_id, target_column, smiles_column, year_column)
        return _skipped(metrics, f"missing {year_column!r} column"), _empty_predictions()

    train_df[year_column] = pd.to_numeric(train_df[year_column], errors="coerce")
    train_df = train_df.dropna(subset=[target_column, year_column]).reset_index(drop=True)
    train_df[year_column] = train_df[year_column].astype(int)

    metrics = _base_metrics(dataset_id, target_column, smiles_column, year_column)
    metrics["eligible_rows"] = int(len(train_df))
    metrics["available_years"] = sorted(train_df[year_column].astype(int).unique().tolist())
    if len(train_df) < 2:
        return _skipped(metrics, "fewer than two temporally eligible verified rows"), _empty_predictions()
    if len(metrics["available_years"]) < 2:
        return _skipped(metrics, "fewer than two distinct verified reference years"), _empty_predictions()

    split = _latest_year_split(
        train_df,
        year_column=year_column,
        validation_fraction=validation_fraction,
        min_validation_rows=min_validation_rows,
    )
    if split is None:
        return _skipped(metrics, "latest-year split produced an empty train or validation side"), _empty_predictions()

    features = feature_fn(train_df[smiles_column])
    feature_frame, feature_columns = _as_feature_frame(features)
    y = train_df[target_column].to_numpy(dtype=float)
    X = feature_frame.to_numpy(dtype=float)
    train_idx = np.flatnonzero(split.train_mask.to_numpy())
    validation_idx = np.flatnonzero(split.validation_mask.to_numpy())

    fitted_model = _fresh_model(model)
    fitted_model.fit(X[train_idx], y[train_idx])
    y_pred = np.asarray(fitted_model.predict(X[validation_idx]), dtype=float)
    y_true = y[validation_idx]
    residual = y_pred - y_true
    validation_rows = train_df.iloc[validation_idx].copy()

    train_dois = _doi_set(train_df.iloc[train_idx])
    validation_dois = _doi_set(validation_rows)
    doi_overlap = sorted(train_dois & validation_dois)
    predictions = pd.DataFrame(
        {
            "record_id": validation_rows.get("record_id", pd.Series(validation_rows.index)).astype(str).to_list(),
            "smiles": validation_rows[smiles_column].astype(str).to_list(),
            "year": validation_rows[year_column].astype(int).to_list(),
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": residual,
            "split": ["validation"] * len(validation_idx),
        },
        columns=PREDICTION_COLUMNS,
    )
    return {
        **metrics,
        "status": "passed",
        "validation_fraction": validation_fraction,
        "min_validation_rows": min_validation_rows,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "feature_columns": feature_columns,
        "model_class": fitted_model.__class__.__name__,
        "train_years": split.train_years,
        "validation_years": split.validation_years,
        "cutoff_year": min(split.validation_years),
        "max_train_year": max(split.train_years),
        "min_validation_year": min(split.validation_years),
        "doi_overlap_count": int(len(doi_overlap)),
        "overlap_dois": doi_overlap,
        "validation_r2": _safe_r2(y_true, y_pred),
        "validation_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "validation_mae": float(mean_absolute_error(y_true, y_pred)),
        "validation_pearson_r": _safe_pearson(y_true, y_pred),
    }, predictions


def _base_metrics(
    dataset_id: str,
    target_column: str,
    smiles_column: str,
    year_column: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "status": "pending",
        "validation_scope": VALIDATION_SCOPE,
        "split_strategy": SPLIT_STRATEGY,
        "external_validation": False,
        "independent_external_validation": False,
        "random_holdout": False,
        "temporal_generalization_diagnostic": True,
        "temporal_generalization_claim": "not_established",
        "target_column": target_column,
        "smiles_column": smiles_column,
        "year_column": year_column,
        "eligible_rows": 0,
        "available_years": [],
        "train_rows": 0,
        "validation_rows": 0,
        "feature_columns": [],
        "model_class": None,
        "train_years": [],
        "validation_years": [],
        "cutoff_year": None,
        "max_train_year": None,
        "min_validation_year": None,
        "doi_overlap_count": 0,
        "overlap_dois": [],
        "validation_r2": None,
        "validation_rmse": None,
        "validation_mae": None,
        "validation_pearson_r": None,
    }


def _latest_year_split(
    train_df: pd.DataFrame,
    *,
    year_column: str,
    validation_fraction: float,
    min_validation_rows: int,
) -> TemporalSplit | None:
    required_validation_rows = max(min_validation_rows, math.ceil(len(train_df) * validation_fraction))
    validation_years: list[int] = []
    validation_mask = pd.Series(False, index=train_df.index)
    for year in sorted(train_df[year_column].astype(int).unique(), reverse=True):
        validation_years.append(int(year))
        validation_mask = train_df[year_column].isin(validation_years)
        train_mask = ~validation_mask
        if int(validation_mask.sum()) >= required_validation_rows and int(train_mask.sum()) > 0:
            train_years = sorted(train_df.loc[train_mask, year_column].astype(int).unique().tolist())
            validation_years_sorted = sorted(validation_years)
            if max(train_years) < min(validation_years_sorted):
                return TemporalSplit(train_mask, validation_mask, train_years, validation_years_sorted)
    return None


def _skipped(metrics: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = dict(metrics)
    payload["status"] = "skipped"
    payload["skip_reason"] = reason
    return payload


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _doi_set(df: pd.DataFrame) -> set[str]:
    if "doi" not in df.columns:
        return set()
    return {str(value).strip().lower() for value in df["doi"].dropna() if str(value).strip()}


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
