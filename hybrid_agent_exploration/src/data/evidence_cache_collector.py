"""Collect missing external evidence cache entries from preflight requirements."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from harness.authenticity import MoleculeEvidence, ReferenceEvidence


REFERENCE_CACHE_FILE = "reference_cache.json"
MOLECULE_CACHE_FILE = "molecule_cache.json"
ReferenceResolver = Callable[[str], ReferenceEvidence | None]
MoleculeResolver = Callable[[dict[str, Any]], MoleculeEvidence | None]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class CacheRequirement:
    """One missing DOI or molecule cache key to resolve."""

    entity_type: str
    key: str
    key_source: str
    row_count: int
    record_ids: list[str]


def collect_evidence_cache(
    *,
    requirements_csv: str | Path,
    cache_dir: str | Path,
    max_requests: int,
    reference_resolver: ReferenceResolver,
    molecule_resolver: MoleculeResolver,
    dataset_id: str | None = None,
    entity_type: str = "all",
    dry_run: bool = False,
    retry_attempts: int = 1,
    include_smiles: bool = False,
    write_negative_cache: bool = False,
    progress_every: int = 0,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Fill missing cache entries with a bounded, resumable collection run."""

    if max_requests < 0:
        raise ValueError("max_requests must be >= 0")
    if retry_attempts < 1:
        raise ValueError("retry_attempts must be >= 1")
    if progress_every < 0:
        raise ValueError("progress_every must be >= 0")
    cache_root = Path(cache_dir)
    reference_cache_path = cache_root / REFERENCE_CACHE_FILE
    molecule_cache_path = cache_root / MOLECULE_CACHE_FILE
    reference_cache = _load_cache(reference_cache_path)
    molecule_cache = _load_cache(molecule_cache_path)
    requirements = _load_requirements(requirements_csv, entity_type=entity_type)
    missing = _select_missing(
        requirements, reference_cache=reference_cache, molecule_cache=molecule_cache
    )
    initial_missing_count = len(missing)
    supported, unsupported = _split_supported(missing, include_smiles=include_smiles)
    planned = supported[:max_requests]

    summary = {
        "schema_version": "evidence-cache-collector-v1",
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "requirements_csv": str(Path(requirements_csv)),
        "cache_dir": str(cache_root),
        "dry_run": dry_run,
        "entity_type": entity_type,
        "max_requests": max_requests,
        "retry_attempts": retry_attempts,
        "write_negative_cache": write_negative_cache,
        "planned_count": len(planned),
        "attempted_count": 0,
        "remaining_planned_count": len(planned),
        "positive_written_count": 0,
        "negative_written_count": 0,
        "no_evidence_count": 0,
        "error_count": 0,
        "unsupported_count": len(unsupported),
        "skipped_existing_count": len(requirements) - len(missing),
        "remaining_missing_count": len(missing),
        "entity_type_counts": {},
        "processed": [],
        "errors": [],
        "unsupported": [
            _requirement_summary(requirement) for requirement in unsupported
        ],
    }
    if dry_run or max_requests == 0:
        summary["remaining_missing_count"] = len(missing)
        summary["entity_type_counts"] = _entity_counts(planned)
        return summary

    for requirement in planned:
        evidence: ReferenceEvidence | MoleculeEvidence | None
        error: str | None
        if requirement.entity_type == "reference":
            evidence, error = _resolve_with_retry(
                lambda: reference_resolver(requirement.key), retry_attempts
            )
            if error is not None:
                _record_error(
                    summary,
                    requirement,
                    error,
                    initial_missing_count=initial_missing_count,
                )
                _emit_progress(
                    summary,
                    progress_every=progress_every,
                    progress_callback=progress_callback,
                )
                continue
            should_write_negative = write_negative_cache
            if evidence or should_write_negative:
                reference_cache[requirement.key] = (
                    evidence.to_source() if evidence else None
                )
                _write_cache(reference_cache, reference_cache_path)
        elif requirement.entity_type == "molecule":
            record = _molecule_record(requirement.key)
            evidence, error = _resolve_with_retry(
                lambda: molecule_resolver(record), retry_attempts
            )
            if error is not None:
                _record_error(
                    summary,
                    requirement,
                    error,
                    initial_missing_count=initial_missing_count,
                )
                _emit_progress(
                    summary,
                    progress_every=progress_every,
                    progress_callback=progress_callback,
                )
                continue
            should_write_negative = write_negative_cache and not _is_smiles_requirement(
                requirement
            )
            if evidence or should_write_negative:
                molecule_cache[requirement.key] = (
                    evidence.to_source() if evidence else None
                )
                _write_cache(molecule_cache, molecule_cache_path)
        else:
            raise ValueError(f"Unsupported entity_type: {requirement.entity_type}")

        summary["attempted_count"] += 1
        if evidence:
            summary["positive_written_count"] += 1
            status = "positive"
        elif should_write_negative:
            summary["negative_written_count"] += 1
            status = "negative"
        else:
            summary["no_evidence_count"] += 1
            status = "no_evidence"
        summary["processed"].append(
            {
                "entity_type": requirement.entity_type,
                "key": requirement.key,
                "cache_status": status,
                "row_count": requirement.row_count,
                "record_ids": requirement.record_ids,
            }
        )
        _update_progress(summary, initial_missing_count=initial_missing_count)
        _emit_progress(
            summary, progress_every=progress_every, progress_callback=progress_callback
        )

    remaining = _select_missing(
        requirements, reference_cache=reference_cache, molecule_cache=molecule_cache
    )
    summary["remaining_planned_count"] = 0
    summary["remaining_missing_count"] = len(remaining)
    summary["entity_type_counts"] = _entity_counts(summary["processed"])
    return summary


def write_collection_report(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Write the collector summary JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _load_requirements(path: str | Path, *, entity_type: str) -> list[CacheRequirement]:
    df = pd.read_csv(path)
    requirements: list[CacheRequirement] = []
    for record in df.to_dict(orient="records"):
        current_entity = _text(record.get("entity_type"))
        if entity_type != "all" and current_entity != entity_type:
            continue
        if _text(record.get("cache_status")) != "missing":
            continue
        requirements.append(
            CacheRequirement(
                entity_type=current_entity,
                key=_text(record.get("key")),
                key_source=_text(record.get("key_source")),
                row_count=_int(record.get("row_count")),
                record_ids=_record_ids(record),
            )
        )
    return [requirement for requirement in requirements if requirement.key]


def _select_missing(
    requirements: Iterable[CacheRequirement],
    *,
    reference_cache: dict[str, Any],
    molecule_cache: dict[str, Any],
) -> list[CacheRequirement]:
    selected: list[CacheRequirement] = []
    for requirement in requirements:
        cache = (
            reference_cache
            if requirement.entity_type == "reference"
            else molecule_cache
        )
        if requirement.key not in cache:
            selected.append(requirement)
    return selected


def _split_supported(
    requirements: Iterable[CacheRequirement],
    *,
    include_smiles: bool,
) -> tuple[list[CacheRequirement], list[CacheRequirement]]:
    supported: list[CacheRequirement] = []
    unsupported: list[CacheRequirement] = []
    for requirement in requirements:
        if requirement.entity_type != "molecule":
            supported.append(requirement)
            continue
        if requirement.key.startswith("pubchem:") or include_smiles:
            supported.append(requirement)
        else:
            unsupported.append(requirement)
    return supported, unsupported


def _resolve_with_retry(
    resolver: Callable[[], ReferenceEvidence | MoleculeEvidence | None],
    retry_attempts: int,
) -> tuple[ReferenceEvidence | MoleculeEvidence | None, str | None]:
    last_error: Exception | None = None
    for _ in range(retry_attempts):
        try:
            return resolver(), None
        except Exception as exc:  # noqa: BLE001 - report transient resolver failures without poisoning caches.
            last_error = exc
    assert last_error is not None
    return None, str(last_error)


def _record_error(
    summary: dict[str, Any],
    requirement: CacheRequirement,
    error: str,
    *,
    initial_missing_count: int,
) -> None:
    summary["attempted_count"] += 1
    summary["error_count"] += 1
    summary["errors"].append({**_requirement_summary(requirement), "error": error})
    _update_progress(summary, initial_missing_count=initial_missing_count)


def _update_progress(summary: dict[str, Any], *, initial_missing_count: int) -> None:
    planned_count = int(summary["planned_count"])
    attempted_count = int(summary["attempted_count"])
    resolved_missing_count = int(summary["positive_written_count"]) + int(
        summary["negative_written_count"]
    )
    summary["remaining_planned_count"] = max(planned_count - attempted_count, 0)
    summary["remaining_missing_count"] = max(
        initial_missing_count - resolved_missing_count, 0
    )
    summary["entity_type_counts"] = _entity_counts(summary["processed"])


def _emit_progress(
    summary: dict[str, Any],
    *,
    progress_every: int,
    progress_callback: ProgressCallback | None,
) -> None:
    if progress_callback is None or progress_every <= 0:
        return
    attempted_count = int(summary["attempted_count"])
    planned_count = int(summary["planned_count"])
    if attempted_count <= 0:
        return
    if attempted_count % progress_every != 0 and attempted_count != planned_count:
        return
    progress_callback(_summary_snapshot(summary))


def _summary_snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(summary)


def _requirement_summary(requirement: CacheRequirement) -> dict[str, Any]:
    return {
        "entity_type": requirement.entity_type,
        "key": requirement.key,
        "row_count": requirement.row_count,
        "record_ids": requirement.record_ids,
    }


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evidence cache must be a JSON object: {path}")
    return payload


def _write_cache(cache: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _molecule_record(key: str) -> dict[str, Any]:
    if key.startswith("pubchem:"):
        return {"pubchem_id": key.removeprefix("pubchem:")}
    if key.startswith("smiles:"):
        return {"smiles": key.removeprefix("smiles:")}
    return {"smiles": key}


def _is_smiles_requirement(requirement: CacheRequirement) -> bool:
    return requirement.entity_type == "molecule" and requirement.key.startswith(
        "smiles:"
    )


def _record_ids(record: dict[str, Any]) -> list[str]:
    payload = _text(record.get("record_ids_json"))
    if payload:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    text = _text(record.get("record_ids"))
    return [item for item in text.split(";") if item]


def _entity_counts(items: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        entity = (
            item.entity_type
            if isinstance(item, CacheRequirement)
            else item["entity_type"]
        )
        counts[entity] = counts.get(entity, 0) + 1
    return counts


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _int(value: Any) -> int:
    text = _text(value)
    if not text:
        return 0
    return int(float(text))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
