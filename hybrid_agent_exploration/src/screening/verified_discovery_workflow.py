"""End-to-end verified candidate discovery workflow.

The workflow connects three gates:
1. strict real-data verification and quarantine,
2. model training on verified rows only,
3. candidate ranking on verified candidate pools only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.verified_dataset_builder import ARTIFACT_POLICY, VerifiedDatasetBuilder
from harness.authenticity import RealDataAuthenticator
from screening.verified_candidate_discovery import (
    CandidateDiscoveryArtifacts,
    VerifiedCandidateDiscovery,
)


FeatureFn = Callable[[pd.Series], Any]


class InsufficientVerifiedDataError(ValueError):
    """Raised when strict verification leaves too few rows to train."""


@dataclass(frozen=True)
class VerifiedDiscoveryWorkflowArtifacts:
    """All outputs emitted by the verified discovery workflow."""

    output_dir: Path
    verified_train_csv: Path
    quarantine_csv: Path
    candidate_pool_csv: Path
    doi_manifest_json: Path
    provenance_json: Path
    data_audit_report_md: Path
    model_metrics_json: Path
    model_manifest_json: Path
    ranked_candidates_csv: Path
    discovery_manifest_json: Path
    discovery_audit_report_md: Path
    workflow_manifest_json: Path
    verified_rows: int
    quarantine_rows: int
    ranked_candidates: int


class VerifiedDiscoveryWorkflow:
    """Run verified data building, training, and candidate ranking."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        authenticator: RealDataAuthenticator | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.authenticator = authenticator

    def run(
        self,
        input_path: str | Path,
        *,
        dataset_id: str | None = None,
        candidate_pool: pd.DataFrame | str | Path | None = None,
        model: Any | None = None,
        feature_fn: FeatureFn | None = None,
        top_k: int = 100,
        min_verified_rows: int = 10,
        target_column: str = "delta_pce",
        smiles_column: str = "smiles",
    ) -> VerifiedDiscoveryWorkflowArtifacts:
        """Run the workflow from a CSV/XLSX input table."""

        path = Path(input_path)
        df = _load_table(path)
        return self.run_from_dataframe(
            df,
            dataset_id=dataset_id or path.stem,
            candidate_pool=candidate_pool,
            model=model,
            feature_fn=feature_fn,
            top_k=top_k,
            min_verified_rows=min_verified_rows,
            target_column=target_column,
            smiles_column=smiles_column,
        )

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        dataset_id: str,
        candidate_pool: pd.DataFrame | str | Path | None = None,
        model: Any | None = None,
        feature_fn: FeatureFn | None = None,
        top_k: int = 100,
        min_verified_rows: int = 10,
        target_column: str = "delta_pce",
        smiles_column: str = "smiles",
    ) -> VerifiedDiscoveryWorkflowArtifacts:
        """Run the full verified discovery workflow from an in-memory table."""

        dataset_dir = self.output_dir / "dataset"
        model_dir = self.output_dir / "model"
        discovery_dir = self.output_dir / "discovery"
        model_dir.mkdir(parents=True, exist_ok=True)

        dataset_artifacts = VerifiedDatasetBuilder(
            self.authenticator,
            output_dir=dataset_dir,
        ).build_from_dataframe(df, dataset_id=dataset_id)

        verified = pd.read_csv(dataset_artifacts.verified_train_csv, low_memory=False)
        _require_training_data(
            verified,
            min_verified_rows=min_verified_rows,
            target_column=target_column,
            smiles_column=smiles_column,
        )

        model = model or RandomForestRegressor(n_estimators=100, random_state=42)
        feature_fn = feature_fn or default_smiles_features
        trained_model, metrics, feature_columns = _train_model(
            verified,
            model=model,
            feature_fn=feature_fn,
            target_column=target_column,
            smiles_column=smiles_column,
        )

        metrics_payload = {
            "dataset_id": dataset_id,
            "generated_at": _now_iso(),
            "train_rows": int(len(verified)),
            "target_column": target_column,
            "smiles_column": smiles_column,
            "feature_columns": feature_columns,
            "model_class": trained_model.__class__.__name__,
            **metrics,
        }
        model_metrics_json = model_dir / "model_metrics.json"
        model_manifest_json = model_dir / "model_manifest.json"
        _write_json(metrics_payload, model_metrics_json)
        _write_json(
            {
                "dataset_id": dataset_id,
                "generated_at": metrics_payload["generated_at"],
                "model_class": trained_model.__class__.__name__,
                "trained_on": dataset_artifacts.verified_train_csv.relative_to(self.output_dir).as_posix(),
                "target_column": target_column,
                "feature_columns": feature_columns,
                "strict_verified_training_only": True,
            },
            model_manifest_json,
        )

        discovery_input = candidate_pool if candidate_pool is not None else dataset_artifacts.candidate_pool_csv
        discovery_artifacts = VerifiedCandidateDiscovery(output_dir=discovery_dir).discover(
            discovery_input,
            model=trained_model,
            feature_fn=feature_fn,
            top_k=top_k,
            dataset_id=dataset_id,
        )

        workflow_manifest_json = self.output_dir / "workflow_manifest.json"
        artifacts = _workflow_artifacts(
            self.output_dir,
            dataset_artifacts,
            model_metrics_json,
            model_manifest_json,
            discovery_artifacts,
        )
        _write_json(
            _workflow_manifest(
                dataset_id,
                artifacts,
                workflow_manifest_json=workflow_manifest_json,
                min_verified_rows=min_verified_rows,
                top_k=top_k,
                model_class=trained_model.__class__.__name__,
            ),
            workflow_manifest_json,
        )
        return VerifiedDiscoveryWorkflowArtifacts(
            output_dir=self.output_dir,
            workflow_manifest_json=workflow_manifest_json,
            **artifacts,
        )


def default_smiles_features(smiles_series: pd.Series) -> pd.DataFrame:
    """Small deterministic feature set for verified candidate discovery.

    The production project can pass a richer RDKit feature function, but this
    default keeps the workflow runnable in minimal environments.
    """

    values = smiles_series.fillna("").astype(str)
    return pd.DataFrame(
        {
            "smiles_length": values.str.len(),
            "carbon_count": values.str.count("C") + values.str.count("c"),
            "oxygen_count": values.str.count("O") + values.str.count("o"),
            "nitrogen_count": values.str.count("N") + values.str.count("n"),
            "sulfur_count": values.str.count("S") + values.str.count("s"),
            "halogen_count": values.str.count("F") + values.str.count("Cl") + values.str.count("Br") + values.str.count("I"),
            "ring_digit_count": values.str.count(r"\d"),
        }
    )


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input table format: {path.suffix}")


def _require_training_data(
    verified: pd.DataFrame,
    *,
    min_verified_rows: int,
    target_column: str,
    smiles_column: str,
) -> None:
    missing = [column for column in (smiles_column, target_column) if column not in verified.columns]
    if missing:
        raise InsufficientVerifiedDataError(f"Verified training data missing columns: {missing}")
    trainable = verified.dropna(subset=[smiles_column, target_column])
    if len(trainable) < min_verified_rows:
        raise InsufficientVerifiedDataError(
            f"Strict authenticity gate left {len(trainable)} trainable rows; "
            f"min_verified_rows={min_verified_rows}."
        )


def _train_model(
    verified: pd.DataFrame,
    *,
    model: Any,
    feature_fn: FeatureFn,
    target_column: str,
    smiles_column: str,
) -> tuple[Any, dict[str, float], list[str]]:
    train_df = verified.dropna(subset=[smiles_column, target_column]).copy()
    train_df[target_column] = pd.to_numeric(train_df[target_column], errors="coerce")
    train_df = train_df.dropna(subset=[target_column])
    features = feature_fn(train_df[smiles_column])
    feature_frame, feature_columns = _as_feature_frame(features)
    y = train_df[target_column].to_numpy(dtype=float)
    X = feature_frame.to_numpy(dtype=float)
    model.fit(X, y)
    y_pred = np.asarray(model.predict(X), dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    metrics = {
        "train_r2": float(r2_score(y, y_pred)) if len(y) > 1 else 0.0,
        "train_rmse": rmse,
        "train_mae": float(mean_absolute_error(y, y_pred)),
    }
    return model, metrics, feature_columns


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


def _workflow_artifacts(
    output_dir: Path,
    dataset_artifacts,
    model_metrics_json: Path,
    model_manifest_json: Path,
    discovery_artifacts: CandidateDiscoveryArtifacts,
) -> dict[str, Any]:
    return {
        "verified_train_csv": dataset_artifacts.verified_train_csv,
        "quarantine_csv": dataset_artifacts.quarantine_csv,
        "candidate_pool_csv": dataset_artifacts.candidate_pool_csv,
        "doi_manifest_json": dataset_artifacts.doi_manifest_json,
        "provenance_json": dataset_artifacts.provenance_json,
        "data_audit_report_md": dataset_artifacts.audit_report_md,
        "model_metrics_json": model_metrics_json,
        "model_manifest_json": model_manifest_json,
        "ranked_candidates_csv": discovery_artifacts.ranked_candidates_csv,
        "discovery_manifest_json": discovery_artifacts.discovery_manifest_json,
        "discovery_audit_report_md": discovery_artifacts.audit_report_md,
        "verified_rows": dataset_artifacts.verified_count,
        "quarantine_rows": dataset_artifacts.quarantine_count,
        "ranked_candidates": discovery_artifacts.ranked_count,
    }


def _workflow_manifest(
    dataset_id: str,
    artifacts: dict[str, Any],
    *,
    workflow_manifest_json: Path,
    min_verified_rows: int,
    top_k: int,
    model_class: str,
) -> dict[str, Any]:
    outputs = {
        key: value.relative_to(artifacts["verified_train_csv"].parents[1]).as_posix()
        for key, value in artifacts.items()
        if key.endswith(("_csv", "_json", "_md"))
    }
    outputs["workflow_manifest_json"] = workflow_manifest_json.relative_to(
        artifacts["verified_train_csv"].parents[1]
    ).as_posix()
    return {
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "artifact_policy": ARTIFACT_POLICY,
        "strict_verified_training_only": True,
        "verified_candidate_discovery_only": True,
        "min_verified_rows": min_verified_rows,
        "top_k": top_k,
        "model_class": model_class,
        "verified_rows": artifacts["verified_rows"],
        "quarantine_rows": artifacts["quarantine_rows"],
        "ranked_candidates": artifacts["ranked_candidates"],
        "outputs": outputs,
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
