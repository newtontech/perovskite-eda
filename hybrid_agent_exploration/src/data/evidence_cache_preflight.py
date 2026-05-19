"""Plan external evidence-cache coverage before a verified discovery run.

The preflight is intentionally read-only and does not call Crossref, PubChem,
or any other network service. It answers whether an ``external-cached`` run can
complete from existing JSON cache files, and which DOI or molecule keys still
need evidence collection.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


EVIDENCE_CACHE_PREFLIGHT_SCHEMA_VERSION = "evidence-cache-preflight-v1"
REFERENCE_CACHE_FILE = "reference_cache.json"
MOLECULE_CACHE_FILE = "molecule_cache.json"


@dataclass(frozen=True)
class EvidenceCachePreflightArtifacts:
    """Paths emitted by an evidence-cache preflight run."""

    output_dir: Path
    summary_json: Path
    requirements_csv: Path
    report_md: Path


def summarize_evidence_cache_preflight(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    source_name: str,
    cache_dir: str | Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Summarize required DOI and molecule cache keys for a source table."""

    cache_root = Path(cache_dir)
    reference_items, reference_uncacheable = _collect_reference_requirements(df)
    molecule_items, molecule_uncacheable = _collect_molecule_requirements(df)
    reference_cache = _load_cache(cache_root / REFERENCE_CACHE_FILE)
    molecule_cache = _load_cache(cache_root / MOLECULE_CACHE_FILE)

    requirements = _requirement_rows("reference", reference_items, reference_cache)
    requirements.extend(_requirement_rows("molecule", molecule_items, molecule_cache))
    reference_coverage = _coverage_summary(
        cache_path=cache_root / REFERENCE_CACHE_FILE,
        requirements=[row for row in requirements if row["entity_type"] == "reference"],
        uncacheable_row_count=reference_uncacheable,
    )
    molecule_coverage = _coverage_summary(
        cache_path=cache_root / MOLECULE_CACHE_FILE,
        requirements=[row for row in requirements if row["entity_type"] == "molecule"],
        uncacheable_row_count=molecule_uncacheable,
    )
    return {
        "schema_version": EVIDENCE_CACHE_PREFLIGHT_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source_name": source_name,
        "generated_at": _now_iso(),
        "network_access": "not_used",
        "row_count": len(df),
        "max_rows": max_rows,
        "max_rows_is_smoke_only": max_rows is not None,
        "cache_dir": str(cache_root),
        "external_cache_ready": _positive_cache_complete(reference_coverage)
        and _positive_cache_complete(molecule_coverage),
        "all_positive_evidence_cached": reference_coverage["negative_cached_count"] == 0
        and molecule_coverage["negative_cached_count"] == 0
        and reference_coverage["missing_count"] == 0
        and molecule_coverage["missing_count"] == 0,
        "reference_cache": reference_coverage,
        "molecule_cache": molecule_coverage,
        "requirements": requirements,
    }


def write_evidence_cache_preflight_artifacts(
    summary: dict[str, Any],
    output_dir: str | Path,
) -> EvidenceCachePreflightArtifacts:
    """Write JSON, CSV, and Markdown evidence-cache preflight artifacts."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = EvidenceCachePreflightArtifacts(
        output_dir=root,
        summary_json=root / "evidence_cache_preflight.json",
        requirements_csv=root / "evidence_cache_requirements.csv",
        report_md=root / "evidence_cache_preflight.md",
    )
    serializable = dict(summary)
    serializable["outputs"] = {
        "summary_json": artifacts.summary_json.name,
        "requirements_csv": artifacts.requirements_csv.name,
        "report_md": artifacts.report_md.name,
    }
    _write_json(serializable, artifacts.summary_json)
    _write_requirements_csv(serializable["requirements"], artifacts.requirements_csv)
    artifacts.report_md.write_text(_format_markdown(serializable), encoding="utf-8")
    return artifacts


def _collect_reference_requirements(df: pd.DataFrame) -> tuple[dict[str, list[str]], int]:
    items: dict[str, list[str]] = {}
    uncacheable = 0
    for index, record in enumerate(df.to_dict(orient="records")):
        key = _normalize_doi(record.get("doi"))
        if not key:
            uncacheable += 1
            continue
        items.setdefault(key, []).append(_record_id(record, index))
    return items, uncacheable


def _collect_molecule_requirements(df: pd.DataFrame) -> tuple[dict[str, list[str]], int]:
    items: dict[str, list[str]] = {}
    uncacheable = 0
    for index, record in enumerate(df.to_dict(orient="records")):
        key = _molecule_cache_key(record)
        if not key:
            uncacheable += 1
            continue
        items.setdefault(key, []).append(_record_id(record, index))
    return items, uncacheable


def _requirement_rows(
    entity_type: str,
    items: dict[str, list[str]],
    cache: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(items):
        cache_status = _cache_status(key, cache)
        rows.append(
            {
                "entity_type": entity_type,
                "key": key,
                "row_count": len(items[key]),
                "record_ids": items[key],
                "cache_status": cache_status,
                "is_cached": cache_status in {"positive", "negative"},
                "has_positive_evidence": cache_status == "positive",
            }
        )
    return rows


def _coverage_summary(
    *,
    cache_path: Path,
    requirements: list[dict[str, Any]],
    uncacheable_row_count: int,
) -> dict[str, Any]:
    required_count = len(requirements)
    cached_count = sum(1 for row in requirements if row["is_cached"])
    positive_cached_count = sum(1 for row in requirements if row["cache_status"] == "positive")
    negative_cached_count = sum(1 for row in requirements if row["cache_status"] == "negative")
    missing = [row["key"] for row in requirements if row["cache_status"] == "missing"]
    return {
        "cache_path": str(cache_path),
        "cache_exists": cache_path.is_file(),
        "required_count": required_count,
        "cached_count": cached_count,
        "positive_cached_count": positive_cached_count,
        "negative_cached_count": negative_cached_count,
        "missing_count": len(missing),
        "uncacheable_row_count": uncacheable_row_count,
        "cache_hit_fraction": _fraction(cached_count, required_count),
        "positive_evidence_fraction": _fraction(positive_cached_count, required_count),
        "missing_keys": missing,
    }


def _positive_cache_complete(summary: dict[str, Any]) -> bool:
    return summary["missing_count"] == 0 and summary["negative_cached_count"] == 0


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evidence cache must be a JSON object: {path}")
    return payload


def _cache_status(key: str, cache: dict[str, Any]) -> str:
    if key not in cache:
        return "missing"
    return "positive" if cache[key] else "negative"


def _write_requirements_csv(requirements: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for row in requirements:
        output = dict(row)
        output["record_ids"] = ";".join(row["record_ids"])
        output["record_ids_json"] = json.dumps(row["record_ids"], ensure_ascii=False)
        rows.append(output)
    columns = [
        "entity_type",
        "key",
        "row_count",
        "cache_status",
        "is_cached",
        "has_positive_evidence",
        "record_ids",
        "record_ids_json",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _format_markdown(summary: dict[str, Any]) -> str:
    reference = summary["reference_cache"]
    molecule = summary["molecule_cache"]
    lines = [
        "# External Evidence Cache Preflight",
        "",
        f"- Dataset: `{summary['dataset_id']}`",
        f"- Source: `{summary['source_name']}`",
        f"- Rows scanned: {summary['row_count']}",
        f"- Network access: `{summary['network_access']}`",
        f"- Cache directory: `{summary['cache_dir']}`",
        f"- External cache ready: `{summary['external_cache_ready']}`",
        f"- All positive evidence cached: `{summary['all_positive_evidence_cached']}`",
        "",
        "## Reference cache coverage",
        "",
        _coverage_line(reference),
        "",
        "## Molecule cache coverage",
        "",
        _coverage_line(molecule),
        "",
        "## Missing Keys",
        "",
    ]
    lines.extend(_missing_lines("Reference", reference["missing_keys"]))
    lines.extend(_missing_lines("Molecule", molecule["missing_keys"]))
    return "\n".join(lines) + "\n"


def _coverage_line(summary: dict[str, Any]) -> str:
    return (
        f"- Required: {summary['required_count']}; cached: {summary['cached_count']}; "
        f"positive: {summary['positive_cached_count']}; negative: {summary['negative_cached_count']}; "
        f"missing: {summary['missing_count']}; uncacheable rows: {summary['uncacheable_row_count']}."
    )


def _missing_lines(label: str, keys: list[str]) -> list[str]:
    if not keys:
        return [f"- {label}: none"]
    return [f"- {label}: `{key}`" for key in keys]


def _normalize_doi(value: Any) -> str:
    return _clean_text(value).lower()


def _molecule_cache_key(record: dict[str, Any]) -> str:
    pubchem_id = _normalize_pubchem_id(record.get("pubchem_id"))
    if pubchem_id:
        return f"pubchem:{pubchem_id}"
    smiles = _clean_text(record.get("smiles"))
    if smiles:
        return f"smiles:{smiles}"
    return ""


def _normalize_pubchem_id(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        parsed = float(text)
    except ValueError:
        return text
    if math.isnan(parsed):
        return ""
    if parsed.is_integer():
        return str(int(parsed))
    return text


def _record_id(record: dict[str, Any], zero_based_index: int) -> str:
    text = _clean_text(record.get("record_id"))
    return text or f"source_row_{zero_based_index + 1:06d}"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 6)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
