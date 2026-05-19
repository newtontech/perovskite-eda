"""Ingest verified discovery artifacts for report provenance sidecars."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DISCOVERY_DIR_KEY = "verified_discovery_artifact_dir"
DISCOVERY_TOP_N_KEY = "verified_discovery_top_n"
DEFAULT_TOP_N = 10


def load_verified_discovery_summary(artifacts: dict[str, Any]) -> dict[str, Any] | None:
    """Load a bounded report summary from a verified discovery artifact dir."""

    artifact_dir_value = artifacts.get(DISCOVERY_DIR_KEY)
    if not artifact_dir_value:
        return None

    artifact_dir = Path(artifact_dir_value)
    top_n = _positive_int(artifacts.get(DISCOVERY_TOP_N_KEY), DEFAULT_TOP_N)
    workflow_manifest = _read_json(artifact_dir / "workflow_manifest.json")
    doi_manifest_path = _manifest_output_path(
        artifact_dir,
        workflow_manifest,
        "doi_manifest_json",
        "dataset/doi_manifest.json",
    )
    ranked_candidates_path = _manifest_output_path(
        artifact_dir,
        workflow_manifest,
        "ranked_candidates_csv",
        "discovery/ranked_candidates.csv",
    )
    quarantine_path = _manifest_output_path(
        artifact_dir,
        workflow_manifest,
        "quarantine_csv",
        "dataset/quarantine.csv",
    )
    doi_manifest = _read_json(doi_manifest_path)
    top_candidates = _read_top_candidates(
        ranked_candidates_path,
        top_n=top_n,
    )
    quarantine_reason_summary = _read_quarantine_reason_summary(quarantine_path)

    return {
        "source_dir": str(artifact_dir),
        "dataset_id": workflow_manifest.get("dataset_id"),
        "artifact_policy": workflow_manifest.get("artifact_policy"),
        "verified_rows": workflow_manifest.get("verified_rows"),
        "quarantine_rows": workflow_manifest.get("quarantine_rows"),
        "ranked_candidates": workflow_manifest.get("ranked_candidates"),
        "top_k": workflow_manifest.get("top_k"),
        "doi_reference_count": doi_manifest.get("reference_count", len(doi_manifest.get("references", []))),
        "top_candidate_limit": top_n,
        "top_candidates": top_candidates,
        "quarantine_reason_summary": dict(sorted(quarantine_reason_summary.items())),
        "artifacts": {
            "workflow_manifest_json": "workflow_manifest.json",
            "ranked_candidates_csv": _relative_to_artifact_dir(ranked_candidates_path, artifact_dir),
            "doi_manifest_json": _relative_to_artifact_dir(doi_manifest_path, artifact_dir),
            "quarantine_csv": _relative_to_artifact_dir(quarantine_path, artifact_dir),
        },
    }


def format_verified_discovery_markdown(summary: dict[str, Any] | None) -> str:
    """Render bounded verified discovery provenance for reports and SI."""

    if not summary:
        return "Verified discovery artifact directory was not supplied."

    lines = [
        f"Artifact source: `{summary.get('source_dir')}`.",
        (
            f"Dataset `{summary.get('dataset_id')}` contained "
            f"{summary.get('verified_rows')} verified rows and "
            f"{summary.get('quarantine_rows')} quarantined rows before candidate ranking."
        ),
        f"DOI references in manifest: {summary.get('doi_reference_count')}.",
        "",
        "### Top Verified Candidates",
        "",
        "| Rank | Record ID | SMILES | Predicted Delta PCE | Uncertainty | DOI |",
        "|------|-----------|--------|---------------------|-------------|-----|",
    ]
    candidates = summary.get("top_candidates", [])
    if candidates:
        for row in candidates:
            lines.append(
                "| {rank} | {record_id} | `{smiles}` | {score} | {uncertainty} | {doi} |".format(
                    rank=_text(row.get("rank")) or "?",
                    record_id=_text(row.get("record_id")) or "?",
                    smiles=_text(row.get("smiles")) or "?",
                    score=_format_float(row.get("predicted_delta_pce")),
                    uncertainty=_format_float(row.get("uncertainty")),
                    doi=_text(row.get("doi")) or "N/A",
                )
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.extend(["", "### Quarantine Reason Summary", ""])
    reasons = summary.get("quarantine_reason_summary", {})
    if reasons:
        lines.extend(f"- {reason}: {count}" for reason, count in sorted(reasons.items()))
    else:
        lines.append("- None")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Verified discovery artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_output_path(
    artifact_dir: Path,
    workflow_manifest: dict[str, Any],
    key: str,
    fallback: str,
) -> Path:
    outputs = workflow_manifest.get("outputs", {})
    value = outputs.get(key, fallback) if isinstance(outputs, dict) else fallback
    path = Path(str(value))
    if path.is_absolute():
        return path
    return artifact_dir / path


def _relative_to_artifact_dir(path: Path, artifact_dir: Path) -> str:
    try:
        return path.relative_to(artifact_dir).as_posix()
    except ValueError:
        return str(path)


def _read_top_candidates(path: Path, *, top_n: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Verified discovery artifact missing: {path}")
    columns = ("rank", "record_id", "smiles", "predicted_delta_pce", "uncertainty", "doi", "verification_status")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({column: _coerce_cell(row.get(column, "")) for column in columns if column in row})
            if len(rows) >= top_n:
                break
    return rows


def _read_quarantine_reason_summary(path: Path) -> Counter[str]:
    if not path.exists():
        raise FileNotFoundError(f"Verified discovery artifact missing: {path}")
    reasons: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for reason in _text(row.get("quarantine_reason")).split(";"):
                reason = reason.strip()
                if reason:
                    reasons[reason] += 1
    return reasons


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_cell(value: Any) -> Any:
    text = _text(value)
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and "." not in text:
        return int(number)
    return number


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
