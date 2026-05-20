"""Ingest real closed-loop experimental feedback as separate artifacts.

The closed-loop ``feedback_template.csv`` remains a placeholder contract. This
module validates user-supplied measurement CSVs into a separate artifact bundle
so package reports can cite feedback without mutating the pending template.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


INGESTION_SCHEMA_VERSION = "closed-loop-feedback-ingestion-manifest-v1"
VALIDATION_SCHEMA_VERSION = "closed-loop-feedback-validation-report-v1"
DEFAULT_FEEDBACK_DIRNAME = "experimental_feedback"
DELTA_TOLERANCE = 1e-6
MIN_ABSOLUTE_PCE = 0.0
MAX_ABSOLUTE_PCE = 100.0
REQUIRED_COLUMNS = (
    "iteration_id",
    "candidate_id",
    "smiles",
    "experimental_status",
    "measured_pce_without_modulator",
    "measured_pce_with_modulator",
    "measured_delta_pce",
    "measurement_date",
    "protocol_reference",
    "lab_or_operator",
    "instrument_or_batch_id",
    "raw_data_reference",
    "data_source",
)
OPTIONAL_COLUMNS = ("notes",)
NUMERIC_COLUMNS = (
    "measured_pce_without_modulator",
    "measured_pce_with_modulator",
    "measured_delta_pce",
)
TEXT_PROVENANCE_COLUMNS = (
    "protocol_reference",
    "lab_or_operator",
    "instrument_or_batch_id",
    "raw_data_reference",
    "data_source",
)
EXTERNAL_LAB_FORBIDDEN_TOKENS = (
    "fixture",
    "mock",
    "synthetic",
    "simulated",
    "generated",
    "demo",
    "example",
    "test",
)


class ExperimentalFeedbackIngestionError(ValueError):
    """Raised when an experimental feedback CSV is incomplete or dishonest."""


@dataclass(frozen=True)
class ExperimentalFeedbackIngestionArtifacts:
    """Paths emitted by experimental feedback ingestion."""

    output_dir: Path
    validated_feedback_csv: Path
    feedback_vs_prediction_csv: Path
    feedback_ingestion_manifest_json: Path
    feedback_validation_report_json: Path

    @property
    def required_files(self) -> tuple[Path, ...]:
        return (
            self.validated_feedback_csv,
            self.feedback_vs_prediction_csv,
            self.feedback_ingestion_manifest_json,
            self.feedback_validation_report_json,
        )


def ingest_experimental_feedback(
    *,
    iteration_dir: str | Path,
    feedback_csv: str | Path,
    output_dir: str | Path | None = None,
    feedback_data_scope: str = "external_lab",
) -> ExperimentalFeedbackIngestionArtifacts:
    """Validate a user-supplied measurement CSV against selected candidates."""

    root = Path(iteration_dir)
    selected_path = root / "selected_candidates.csv"
    template_path = root / "feedback_template.csv"
    feedback_path = Path(feedback_csv)
    _validate_input_path(feedback_path, iteration_dir=root)
    selected = _read_selected_candidates(selected_path)
    feedback = _read_feedback(feedback_path)
    validated = _validate_feedback_rows(
        feedback,
        selected,
        feedback_csv=feedback_path,
        feedback_data_scope=feedback_data_scope,
    )
    comparison = _feedback_vs_prediction(validated, selected)

    artifacts = ExperimentalFeedbackIngestionArtifacts(
        output_dir=Path(output_dir) if output_dir is not None else root / DEFAULT_FEEDBACK_DIRNAME,
        validated_feedback_csv=(
            (Path(output_dir) if output_dir is not None else root / DEFAULT_FEEDBACK_DIRNAME)
            / "validated_feedback.csv"
        ),
        feedback_vs_prediction_csv=(
            (Path(output_dir) if output_dir is not None else root / DEFAULT_FEEDBACK_DIRNAME)
            / "feedback_vs_prediction.csv"
        ),
        feedback_ingestion_manifest_json=(
            (Path(output_dir) if output_dir is not None else root / DEFAULT_FEEDBACK_DIRNAME)
            / "feedback_ingestion_manifest.json"
        ),
        feedback_validation_report_json=(
            (Path(output_dir) if output_dir is not None else root / DEFAULT_FEEDBACK_DIRNAME)
            / "feedback_validation_report.json"
        ),
    )
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    validated.to_csv(artifacts.validated_feedback_csv, index=False)
    comparison.to_csv(artifacts.feedback_vs_prediction_csv, index=False)
    selected_ids = set(selected["candidate_id"].astype(str))
    ingested_ids = set(validated["candidate_id"].astype(str))
    missing_selected_ids = sorted(selected_ids - ingested_ids)
    _write_json(
        _ingestion_manifest(
            artifacts,
            feedback_csv=feedback_path,
            selected_candidates_csv=selected_path,
            feedback_template_csv=template_path,
            feedback_data_scope=feedback_data_scope,
            selected_rows=len(selected),
            raw_feedback_rows=len(feedback),
            validated_rows=len(validated),
            missing_selected_ids=missing_selected_ids,
        ),
        artifacts.feedback_ingestion_manifest_json,
    )
    _write_json(
        _validation_report(
            feedback_data_scope=feedback_data_scope,
            selected_candidate_ids=sorted(selected_ids),
            ingested_candidate_ids=sorted(ingested_ids),
            missing_selected_candidate_ids=missing_selected_ids,
            validated_rows=len(validated),
        ),
        artifacts.feedback_validation_report_json,
    )
    return artifacts


def _validate_input_path(path: Path, *, iteration_dir: Path) -> None:
    if path.suffix.lower() != ".csv":
        raise ExperimentalFeedbackIngestionError("experimental feedback input must be a .csv file")
    if not path.is_file():
        raise ExperimentalFeedbackIngestionError(f"experimental feedback CSV missing: {path}")
    if path.is_symlink():
        raise ExperimentalFeedbackIngestionError("experimental feedback CSV must not be a symlink")
    try:
        path.resolve().relative_to(iteration_dir.resolve())
    except ValueError:
        pass
    else:
        raise ExperimentalFeedbackIngestionError(
            "experimental feedback CSV must be user-supplied outside the generated iteration directory"
        )


def _read_selected_candidates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ExperimentalFeedbackIngestionError(f"selected candidates file missing: {path}")
    selected = pd.read_csv(path, keep_default_na=False)
    for column in ("iteration_id", "candidate_id", "smiles", "predicted_delta_pce"):
        if column not in selected.columns:
            raise ExperimentalFeedbackIngestionError(f"selected_candidates.csv missing required column: {column}")
    if selected.empty:
        raise ExperimentalFeedbackIngestionError("selected_candidates.csv contains no selected candidates")
    return selected


def _read_feedback(path: Path) -> pd.DataFrame:
    feedback = pd.read_csv(path, keep_default_na=False)
    if feedback.empty:
        raise ExperimentalFeedbackIngestionError("experimental feedback CSV contains no rows")
    missing = [column for column in REQUIRED_COLUMNS if column not in feedback.columns]
    if missing:
        raise ExperimentalFeedbackIngestionError(
            "experimental feedback CSV missing required columns: " + ", ".join(missing)
        )
    for column in OPTIONAL_COLUMNS:
        if column not in feedback.columns:
            feedback[column] = ""
    return feedback


def _validate_feedback_rows(
    feedback: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    feedback_csv: Path,
    feedback_data_scope: str,
) -> pd.DataFrame:
    errors: list[str] = []
    if feedback["candidate_id"].astype(str).duplicated().any():
        duplicates = sorted(feedback.loc[feedback["candidate_id"].astype(str).duplicated(), "candidate_id"].astype(str))
        errors.append(f"duplicate candidate_id values: {duplicates}")
    selected_ids = set(selected["candidate_id"].astype(str))
    feedback_ids = set(feedback["candidate_id"].astype(str))
    unknown_ids = sorted(feedback_ids - selected_ids)
    if unknown_ids:
        errors.append(f"unknown candidate_id values not in selected_candidates.csv: {unknown_ids}")
    selected_iterations = set(selected["iteration_id"].astype(str))
    bad_iterations = sorted(set(feedback["iteration_id"].astype(str)) - selected_iterations)
    if bad_iterations:
        errors.append(f"feedback iteration_id values do not match selected candidates: {bad_iterations}")
    bad_statuses = sorted(set(feedback["experimental_status"].astype(str)) - {"measured"})
    if bad_statuses:
        errors.append(f"experimental_status must be 'measured'; observed {bad_statuses}")
    selected_identity = {
        (
            str(row["iteration_id"]),
            str(row["candidate_id"]),
            str(row["smiles"]),
        )
        for row in selected[["iteration_id", "candidate_id", "smiles"]].to_dict(orient="records")
    }
    feedback_identity = {
        (
            str(row["iteration_id"]),
            str(row["candidate_id"]),
            str(row["smiles"]),
        )
        for row in feedback[["iteration_id", "candidate_id", "smiles"]].to_dict(orient="records")
    }
    mismatched_identity = sorted(feedback_identity - selected_identity)
    if mismatched_identity:
        errors.append(
            "feedback rows must match selected candidate identity "
            f"(iteration_id, candidate_id, smiles): {mismatched_identity}"
        )

    for idx, row in feedback.iterrows():
        row_id = str(row.get("candidate_id", f"row-{idx}"))
        for column in TEXT_PROVENANCE_COLUMNS:
            if _text(row.get(column)) == "":
                errors.append(f"{column} is required for candidate_id={row_id}")
        try:
            date.fromisoformat(_text(row.get("measurement_date")))
        except ValueError:
            errors.append(f"measurement_date must be ISO YYYY-MM-DD for candidate_id={row_id}")
        values: dict[str, float] = {}
        for column in NUMERIC_COLUMNS:
            try:
                value = float(row.get(column))
            except (TypeError, ValueError):
                errors.append(f"{column} must be a finite numeric value for candidate_id={row_id}")
                continue
            if not math.isfinite(value):
                errors.append(f"{column} must be a finite numeric value for candidate_id={row_id}")
                continue
            if column in {"measured_pce_without_modulator", "measured_pce_with_modulator"} and not (
                MIN_ABSOLUTE_PCE <= value <= MAX_ABSOLUTE_PCE
            ):
                errors.append(f"{column} must be between 0 and 100 for candidate_id={row_id}")
                continue
            values[column] = value
        if set(NUMERIC_COLUMNS).issubset(values):
            expected_delta = values["measured_pce_with_modulator"] - values["measured_pce_without_modulator"]
            if abs(values["measured_delta_pce"] - expected_delta) > DELTA_TOLERANCE:
                errors.append(
                    "measured_delta_pce must equal measured_pce_with_modulator "
                    f"- measured_pce_without_modulator for candidate_id={row_id}"
                )
    if feedback_data_scope == "external_lab":
        _check_external_lab_fake_markers(feedback, feedback_csv=feedback_csv, errors=errors)
    if errors:
        raise ExperimentalFeedbackIngestionError("; ".join(errors))

    validated = feedback.copy()
    for column in NUMERIC_COLUMNS:
        validated[column] = validated[column].astype(float)
    validated["measurement_date"] = validated["measurement_date"].map(_text)
    validated["feedback_data_scope"] = feedback_data_scope
    validated["ingested_at"] = _now_iso()
    return validated[list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS) + ["feedback_data_scope", "ingested_at"]]


def _check_external_lab_fake_markers(feedback: pd.DataFrame, *, feedback_csv: Path, errors: list[str]) -> None:
    fields = [feedback_csv.name]
    for column in TEXT_PROVENANCE_COLUMNS + ("notes",):
        fields.extend(feedback[column].astype(str).tolist())
    field_tokens = set()
    for field in fields:
        field_tokens.update(token for token in re.split(r"[^a-z0-9]+", field.lower()) if token)
    matches = sorted(set(EXTERNAL_LAB_FORBIDDEN_TOKENS) & field_tokens)
    if matches:
        errors.append(
            "external_lab feedback cannot contain fixture/mock/test markers; "
            f"matched tokens: {matches}"
        )


def _feedback_vs_prediction(validated: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_subset = selected[
        [
            "iteration_id",
            "candidate_id",
            "smiles",
            "rank",
            "predicted_delta_pce",
            "candidate_score",
        ]
    ].copy()
    selected_subset["predicted_delta_pce"] = pd.to_numeric(
        selected_subset["predicted_delta_pce"],
        errors="coerce",
    )
    merged = validated.merge(
        selected_subset,
        on=["iteration_id", "candidate_id", "smiles"],
        how="left",
        validate="one_to_one",
    )
    merged["prediction_error_delta_pce"] = merged["predicted_delta_pce"] - merged["measured_delta_pce"]
    return merged[
        [
            "iteration_id",
            "candidate_id",
            "smiles",
            "rank",
            "candidate_score",
            "predicted_delta_pce",
            "measured_delta_pce",
            "prediction_error_delta_pce",
            "measured_pce_without_modulator",
            "measured_pce_with_modulator",
            "measurement_date",
            "protocol_reference",
            "lab_or_operator",
            "instrument_or_batch_id",
            "raw_data_reference",
            "data_source",
            "notes",
            "feedback_data_scope",
        ]
    ]


def _ingestion_manifest(
    artifacts: ExperimentalFeedbackIngestionArtifacts,
    *,
    feedback_csv: Path,
    selected_candidates_csv: Path,
    feedback_template_csv: Path,
    feedback_data_scope: str,
    selected_rows: int,
    raw_feedback_rows: int,
    validated_rows: int,
    missing_selected_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": INGESTION_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "feedback_data_scope": feedback_data_scope,
        "delta_tolerance": DELTA_TOLERANCE,
        "input_feedback_file": _full_file_facts(feedback_csv),
        "selected_candidates_file": _full_file_facts(selected_candidates_csv),
        "feedback_template_file": _full_file_facts(feedback_template_csv),
        "row_counts": {
            "selected_candidate_rows": selected_rows,
            "raw_feedback_rows": raw_feedback_rows,
            "validated_feedback_rows": validated_rows,
            "missing_selected_candidate_rows": len(missing_selected_ids),
        },
        "missing_selected_candidate_ids": missing_selected_ids,
        "outputs": {
            "validated_feedback_csv": artifacts.validated_feedback_csv.name,
            "feedback_vs_prediction_csv": artifacts.feedback_vs_prediction_csv.name,
            "feedback_ingestion_manifest_json": artifacts.feedback_ingestion_manifest_json.name,
            "feedback_validation_report_json": artifacts.feedback_validation_report_json.name,
        },
    }


def _validation_report(
    *,
    feedback_data_scope: str,
    selected_candidate_ids: list[str],
    ingested_candidate_ids: list[str],
    missing_selected_candidate_ids: list[str],
    validated_rows: int,
) -> dict[str, Any]:
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": "passed",
        "feedback_data_scope": feedback_data_scope,
        "validated_feedback_rows": validated_rows,
        "checks": {
            "candidate_ids": {
                "status": "passed",
                "selected_candidate_ids": selected_candidate_ids,
                "ingested_candidate_ids": ingested_candidate_ids,
                "missing_selected_candidate_ids": missing_selected_candidate_ids,
                "unknown_candidate_ids": [],
            },
            "measurement_fields": {
                "status": "passed",
                "numeric_columns": list(NUMERIC_COLUMNS),
                "delta_tolerance": DELTA_TOLERANCE,
            },
            "measurement_provenance": {
                "status": "passed",
                "required_columns": list(TEXT_PROVENANCE_COLUMNS) + ["measurement_date"],
            },
        },
    }


def _full_file_facts(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text
