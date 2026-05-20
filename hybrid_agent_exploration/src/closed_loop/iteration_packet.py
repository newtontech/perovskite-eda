"""Generate closed-loop iteration handoff packets without fake measurements."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "closed-loop-iteration-packet-v1"
FEEDBACK_SCHEMA_VERSION = "closed-loop-feedback-schema-v1"
VERIFICATION_SCHEMA_VERSION = "closed-loop-verification-report-v1"
REQUIRED_FILENAMES = (
    "iteration_manifest.json",
    "selected_candidates.csv",
    "candidate_packet.md",
    "feedback_schema.json",
    "feedback_template.csv",
    "verification_report.json",
)
MEASUREMENT_COLUMNS = (
    "measured_delta_pce",
    "measured_pce",
    "measured_pce_without_modulator",
    "measured_pce_with_modulator",
    "measurement_date",
    "protocol_reference",
    "notes",
)
SELECTED_COLUMNS = (
    "iteration_id",
    "candidate_id",
    "rank",
    "smiles",
    "predicted_delta_pce",
    "candidate_score",
    "score_components",
    "uncertainty",
    "source_name",
    "source_url",
    "pubchem_id",
    "cas_number",
    "availability_status",
    "synthesis_status",
    "safety_status",
    "verification_status",
    "verification_sources",
)


class ClosedLoopPacketError(ValueError):
    """Raised when a closed-loop handoff packet is incomplete or dishonest."""


@dataclass(frozen=True)
class ClosedLoopIterationArtifacts:
    """Paths emitted by a closed-loop iteration handoff packet."""

    output_dir: Path
    iteration_manifest_json: Path
    selected_candidates_csv: Path
    candidate_packet_md: Path
    feedback_schema_json: Path
    feedback_template_csv: Path
    verification_report_json: Path

    @property
    def required_files(self) -> tuple[Path, ...]:
        return (
            self.iteration_manifest_json,
            self.selected_candidates_csv,
            self.candidate_packet_md,
            self.feedback_schema_json,
            self.feedback_template_csv,
            self.verification_report_json,
        )


def generate_iteration_packet(
    *,
    ranked_candidates_csv: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    iteration_index: int = 0,
    top_k: int = 10,
) -> ClosedLoopIterationArtifacts:
    """Write a closed-loop candidate handoff packet for real lab feedback."""

    ranked_path = Path(ranked_candidates_csv)
    iteration_dir = Path(output_dir)
    iteration_dir.mkdir(parents=True, exist_ok=True)
    if top_k < 1:
        raise ClosedLoopPacketError("top_k must be >= 1")
    ranked = _load_ranked_candidates(ranked_path)
    selected = _select_candidates(ranked, top_k=top_k, iteration_id=_iteration_id(iteration_index))

    artifacts = ClosedLoopIterationArtifacts(
        output_dir=iteration_dir,
        iteration_manifest_json=iteration_dir / "iteration_manifest.json",
        selected_candidates_csv=iteration_dir / "selected_candidates.csv",
        candidate_packet_md=iteration_dir / "candidate_packet.md",
        feedback_schema_json=iteration_dir / "feedback_schema.json",
        feedback_template_csv=iteration_dir / "feedback_template.csv",
        verification_report_json=iteration_dir / "verification_report.json",
    )
    selected.to_csv(artifacts.selected_candidates_csv, index=False)
    artifacts.candidate_packet_md.write_text(
        _candidate_packet_markdown(
            selected,
            dataset_id=dataset_id,
            iteration_index=iteration_index,
            top_k=top_k,
            ranked_candidates_csv=ranked_path,
        ),
        encoding="utf-8",
    )
    _write_json(_feedback_schema(), artifacts.feedback_schema_json)
    _feedback_template(selected).to_csv(artifacts.feedback_template_csv, index=False)
    report = _verification_report(
        iteration_dir,
        ranked,
        selected,
        top_k=top_k,
        assume_manifest_exists=True,
        assume_report_exists=True,
    )
    _write_json(report, artifacts.verification_report_json)
    _write_json(
        _iteration_manifest(
            artifacts,
            dataset_id=dataset_id,
            iteration_index=iteration_index,
            top_k=top_k,
            ranked_candidates_csv=ranked_path,
            selected_count=len(selected),
        ),
        artifacts.iteration_manifest_json,
    )
    return artifacts


def verify_iteration_packet(
    iteration_dir: str | Path,
    *,
    ranked_candidates_csv: str | Path,
    top_k: int,
) -> dict[str, Any]:
    """Verify that a closed-loop packet contains only pending measurement slots."""

    root = Path(iteration_dir)
    ranked = _load_ranked_candidates(Path(ranked_candidates_csv))
    errors: list[str] = []
    missing = [name for name in REQUIRED_FILENAMES if not (root / name).is_file()]
    if missing:
        errors.append(f"missing required closed-loop files: {', '.join(missing)}")
        selected = pd.DataFrame()
    else:
        selected = pd.read_csv(root / "selected_candidates.csv", keep_default_na=False)

    feedback = _read_feedback(root / "feedback_template.csv", errors)
    manifest = _read_json(root / "iteration_manifest.json", errors, context="iteration_manifest.json")
    if isinstance(manifest, dict) and manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            "iteration_manifest.json schema_version="
            f"{manifest.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
        )

    if not selected.empty:
        _check_top_k_subset(ranked, selected, top_k=top_k, errors=errors)
        _check_verified_only(selected, errors=errors)
    _check_measurement_placeholders(feedback, errors=errors)
    _check_hashes_present(manifest, errors=errors)

    report = _verification_report(root, ranked, selected, top_k=top_k, errors=errors)
    if errors:
        raise ClosedLoopPacketError("; ".join(errors))
    return report


def _load_ranked_candidates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ClosedLoopPacketError(f"ranked candidates file missing: {path}")
    ranked = pd.read_csv(path, keep_default_na=False)
    for column in ("candidate_id", "rank", "verification_status"):
        if column not in ranked.columns:
            raise ClosedLoopPacketError(f"ranked candidates missing required column: {column}")
    return ranked


def _select_candidates(ranked: pd.DataFrame, *, top_k: int, iteration_id: str) -> pd.DataFrame:
    sorted_ranked = ranked.copy()
    sorted_ranked["rank"] = pd.to_numeric(sorted_ranked["rank"], errors="coerce")
    sorted_ranked = sorted_ranked.sort_values("rank", kind="mergesort").head(top_k).reset_index(drop=True)
    _check_verified_only(sorted_ranked, errors := [])
    if errors:
        raise ClosedLoopPacketError("; ".join(errors))
    selected = pd.DataFrame()
    selected["iteration_id"] = [iteration_id] * len(sorted_ranked)
    for column in SELECTED_COLUMNS:
        if column == "iteration_id":
            continue
        selected[column] = sorted_ranked[column] if column in sorted_ranked.columns else ""
    selected["rank"] = pd.to_numeric(selected["rank"], errors="coerce").astype("Int64")
    return selected[list(SELECTED_COLUMNS)]


def _feedback_template(selected: pd.DataFrame) -> pd.DataFrame:
    feedback = selected[["iteration_id", "candidate_id", "smiles"]].copy()
    feedback["experimental_status"] = "pending_external_measurement"
    for column in MEASUREMENT_COLUMNS:
        feedback[column] = ""
    return feedback


def _feedback_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "iteration_id": {"type": "string"},
        "candidate_id": {"type": "string"},
        "smiles": {"type": "string"},
        "experimental_status": {"type": "string", "enum": ["pending_external_measurement"]},
    }
    for column in MEASUREMENT_COLUMNS:
        properties[column] = {"type": "string"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": ["iteration_id", "candidate_id", "experimental_status"],
        "properties": properties,
    }


def _candidate_packet_markdown(
    selected: pd.DataFrame,
    *,
    dataset_id: str,
    iteration_index: int,
    top_k: int,
    ranked_candidates_csv: Path,
) -> str:
    lines = [
        f"# Closed-loop candidate packet: {_iteration_id(iteration_index)}",
        "",
        f"Dataset: `{dataset_id}`",
        f"Selection policy: top {top_k} verified rows from `{ranked_candidates_csv}` sorted by rank.",
        "",
        "No experimental measurement has been generated or filled by this packet.",
        "All measurement fields are placeholders for external laboratory feedback.",
        "",
        "| Rank | Candidate ID | SMILES | Predicted ΔPCE | Candidate score |",
        "|---:|---|---|---:|---:|",
    ]
    for _, row in selected.iterrows():
        lines.append(
            "| "
            f"{row.get('rank', '')} | {row.get('candidate_id', '')} | {row.get('smiles', '')} | "
            f"{row.get('predicted_delta_pce', '')} | {row.get('candidate_score', '')} |"
        )
    lines.extend(
        [
            "",
            "Provenance caveat: selected candidates inherit the verification status and source fields from ranked_candidates.csv. "
            "Candidate priority is an internal model-driven handoff artifact, not experimental validation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _iteration_manifest(
    artifacts: ClosedLoopIterationArtifacts,
    *,
    dataset_id: str,
    iteration_index: int,
    top_k: int,
    ranked_candidates_csv: Path,
    selected_count: int,
) -> dict[str, Any]:
    hash_targets = [
        artifacts.selected_candidates_csv,
        artifacts.candidate_packet_md,
        artifacts.feedback_schema_json,
        artifacts.feedback_template_csv,
        artifacts.verification_report_json,
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "iteration_id": _iteration_id(iteration_index),
        "iteration_index": iteration_index,
        "generated_at": _now_iso(),
        "selection_policy": {
            "source": "ranked_candidates_top_k",
            "ranked_candidates_csv": str(ranked_candidates_csv),
            "top_k": top_k,
            "sort_keys": ["rank"],
        },
        "selected_count": int(selected_count),
        "experimental_measurement_policy": {
            "does_not_generate_or_fill_measurements": True,
            "measurement_status": "pending_external_feedback",
        },
        "outputs": {path.name: path.name for path in artifacts.required_files},
        "file_hashes": {path.name: _sha256(path) for path in hash_targets},
    }


def _verification_report(
    iteration_dir: Path,
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    top_k: int,
    errors: list[str] | None = None,
    assume_manifest_exists: bool = False,
    assume_report_exists: bool = False,
) -> dict[str, Any]:
    current_errors = list(errors or [])
    required_missing = [
        name
        for name in REQUIRED_FILENAMES
        if not (
            (iteration_dir / name).is_file()
            or (assume_manifest_exists and name == "iteration_manifest.json")
            or (assume_report_exists and name == "verification_report.json")
        )
    ]
    top_k_passed = not selected.empty and _expected_candidate_ids(ranked, top_k) == selected["candidate_id"].astype(str).tolist()
    verified_passed = not selected.empty and set(selected["verification_status"].astype(str)) == {"verified"}
    feedback = _read_feedback(iteration_dir / "feedback_template.csv", current_errors)
    measurement_errors: list[str] = []
    _check_measurement_placeholders(feedback, errors=measurement_errors)
    if measurement_errors:
        current_errors.extend(measurement_errors)
    status = "failed" if current_errors or required_missing or not top_k_passed or not verified_passed else "passed"
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": status,
        "generated_at": _now_iso(),
        "checks": {
            "required_files": {
                "status": "passed" if not required_missing else "failed",
                "missing": required_missing,
            },
            "top_k_subset": {"status": "passed" if top_k_passed else "failed"},
            "verified_only": {"status": "passed" if verified_passed else "failed"},
            "measurement_placeholders_empty": {
                "status": "passed" if not measurement_errors else "failed",
                "errors": measurement_errors,
            },
            "hashes_present": {
                "status": "passed",
                "policy": "sha256 full hex recorded in iteration_manifest.file_hashes",
            },
        },
    }


def _check_top_k_subset(
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    top_k: int,
    errors: list[str],
) -> None:
    expected = _expected_candidate_ids(ranked, top_k)
    observed = selected["candidate_id"].astype(str).tolist() if "candidate_id" in selected.columns else []
    if observed != expected:
        errors.append(f"selected_candidates.csv candidate_id order {observed} does not match top-k ranked {expected}")


def _expected_candidate_ids(ranked: pd.DataFrame, top_k: int) -> list[str]:
    sorted_ranked = ranked.copy()
    sorted_ranked["rank"] = pd.to_numeric(sorted_ranked["rank"], errors="coerce")
    sorted_ranked = sorted_ranked.sort_values("rank", kind="mergesort").head(top_k)
    return sorted_ranked["candidate_id"].astype(str).tolist()


def _check_verified_only(df: pd.DataFrame, errors: list[str]) -> None:
    if "verification_status" not in df.columns:
        errors.append("verification_status column is required")
        return
    status = df["verification_status"].fillna("").astype(str)
    bad = df[status != "verified"]
    if not bad.empty:
        ids = bad.get("candidate_id", bad.index.to_series()).astype(str).tolist()
        errors.append(f"closed-loop packet requires verified-only candidates; blocked rows: {', '.join(ids)}")


def _check_measurement_placeholders(feedback: pd.DataFrame, *, errors: list[str]) -> None:
    if feedback.empty:
        return
    for column in MEASUREMENT_COLUMNS:
        if column not in feedback.columns:
            errors.append(f"feedback_template.csv missing measurement placeholder column {column}")
            continue
        values = feedback[column].fillna("").astype(str).str.strip()
        if (values != "").any():
            errors.append(f"feedback_template.csv column {column} must be empty until real external feedback exists")
    if "experimental_status" not in feedback.columns:
        errors.append("feedback_template.csv missing experimental_status")
    elif set(feedback["experimental_status"].astype(str)) != {"pending_external_measurement"}:
        errors.append("feedback_template.csv experimental_status must remain pending_external_measurement")


def _check_hashes_present(manifest: Any, *, errors: list[str]) -> None:
    if not isinstance(manifest, dict):
        errors.append("iteration_manifest.json must be a JSON object")
        return
    hashes = manifest.get("file_hashes")
    if not isinstance(hashes, dict):
        errors.append("iteration_manifest.json file_hashes must be an object")
        return
    missing = sorted(set(REQUIRED_FILENAMES) - {"iteration_manifest.json"} - set(hashes))
    if missing:
        errors.append(f"iteration_manifest.json file_hashes missing {missing}")


def _read_feedback(path: Path, errors: list[str]) -> pd.DataFrame:
    if not path.is_file():
        errors.append(f"feedback_template.csv missing at {path}")
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def _read_json(path: Path, errors: list[str], *, context: str) -> Any:
    if not path.is_file():
        errors.append(f"{context} missing at {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{context} invalid JSON: {exc}")
        return None


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iteration_id(iteration_index: int) -> str:
    return f"iteration_{iteration_index:03d}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
