"""Collect missing external evidence cache entries from preflight requirements."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
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
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    """Fill missing cache entries with a bounded, resumable collection run."""

    if max_requests < 0:
        raise ValueError("max_requests must be >= 0")
    if retry_attempts < 1:
        raise ValueError("retry_attempts must be >= 1")
    if progress_every < 0:
        raise ValueError("progress_every must be >= 0")
    _validate_shard(shard_count=shard_count, shard_index=shard_index)
    cache_root = Path(cache_dir)
    reference_cache_path = cache_root / REFERENCE_CACHE_FILE
    molecule_cache_path = cache_root / MOLECULE_CACHE_FILE
    reference_cache = _load_cache(reference_cache_path)
    molecule_cache = _load_cache(molecule_cache_path)
    requirements = _load_requirements(requirements_csv, entity_type=entity_type)
    supported, unsupported = _split_supported(
        requirements, include_smiles=include_smiles
    )
    shard_supported = _select_shard(
        supported, shard_count=shard_count, shard_index=shard_index
    )
    shard_unsupported = _select_shard(
        unsupported, shard_count=shard_count, shard_index=shard_index
    )
    missing = _select_missing(
        shard_supported, reference_cache=reference_cache, molecule_cache=molecule_cache
    )
    unsupported_missing = _select_missing(
        shard_unsupported,
        reference_cache=reference_cache,
        molecule_cache=molecule_cache,
    )
    initial_missing_count = len(missing) + len(unsupported_missing)
    planned = missing[:max_requests]

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
        "shard": {"count": shard_count, "index": shard_index},
        "planned_count": len(planned),
        "attempted_count": 0,
        "remaining_planned_count": len(planned),
        "positive_written_count": 0,
        "negative_written_count": 0,
        "no_evidence_count": 0,
        "error_count": 0,
        "unsupported_count": len(unsupported_missing),
        "skipped_existing_count": (
            len(shard_supported) + len(shard_unsupported) - initial_missing_count
        ),
        "remaining_missing_count": initial_missing_count,
        "entity_type_counts": {},
        "processed": [],
        "planned": [_requirement_summary(requirement) for requirement in planned],
        "errors": [],
        "unsupported": [
            _requirement_summary(requirement) for requirement in unsupported_missing
        ],
    }
    if dry_run or max_requests == 0:
        summary["remaining_missing_count"] = initial_missing_count
        summary["entity_type_counts"] = _entity_counts(planned)
        return summary

    _initialize_cache_files(
        reference_cache=reference_cache,
        reference_cache_path=reference_cache_path,
        molecule_cache=molecule_cache,
        molecule_cache_path=molecule_cache_path,
    )

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
        shard_supported, reference_cache=reference_cache, molecule_cache=molecule_cache
    )
    remaining.extend(
        _select_missing(
            shard_unsupported,
            reference_cache=reference_cache,
            molecule_cache=molecule_cache,
        )
    )
    summary["remaining_planned_count"] = 0
    summary["remaining_missing_count"] = len(remaining)
    summary["entity_type_counts"] = _entity_counts(summary["processed"])
    return summary


def write_collection_report(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Write the collector summary JSON."""

    path = Path(output_path)
    _write_json_atomic(summary, path)
    return path


def merge_evidence_cache_shards(
    *,
    canonical_cache_dir: str | Path,
    shard_cache_dirs: Iterable[str | Path],
    entity_type: str = "all",
    conflict_policy: str = "fail",
) -> dict[str, Any]:
    """Merge shard cache JSON files into the canonical cache.

    The default conflict policy fails before writing canonical files. Use
    ``keep-canonical`` to keep existing canonical entries and earlier shards.
    """

    if entity_type not in {"all", "reference", "molecule"}:
        raise ValueError("entity_type must be one of: all, reference, molecule")
    if conflict_policy not in {"fail", "keep-canonical"}:
        raise ValueError("conflict_policy must be one of: fail, keep-canonical")

    canonical_root = Path(canonical_cache_dir)
    ordered_shards = sorted(
        (Path(path) for path in shard_cache_dirs), key=lambda p: str(p)
    )
    merge_plan: dict[str, tuple[Path, list[Path]]] = {}
    if entity_type in {"all", "reference"}:
        merge_plan["reference"] = (
            canonical_root / REFERENCE_CACHE_FILE,
            [shard / REFERENCE_CACHE_FILE for shard in ordered_shards],
        )
    if entity_type in {"all", "molecule"}:
        merge_plan["molecule"] = (
            canonical_root / MOLECULE_CACHE_FILE,
            [shard / MOLECULE_CACHE_FILE for shard in ordered_shards],
        )

    file_summaries = {
        cache_type: _merge_cache_file(canonical_path, shard_paths)
        for cache_type, (canonical_path, shard_paths) in merge_plan.items()
    }
    conflicts = [
        conflict
        for summary in file_summaries.values()
        for conflict in summary["conflicts"]
    ]
    merge_status = "failed" if conflicts and conflict_policy == "fail" else "merged"
    if merge_status == "merged":
        for cache_type, (canonical_path, _) in merge_plan.items():
            _write_cache(file_summaries[cache_type]["cache"], canonical_path)

    return {
        "schema_version": "evidence-cache-merge-v1",
        "generated_at": _now_iso(),
        "canonical_cache_dir": str(canonical_root),
        "shard_cache_dirs": [str(path) for path in ordered_shards],
        "entity_type": entity_type,
        "conflict_policy": conflict_policy,
        "merge_status": merge_status,
        "merged_counts": {
            "reference": file_summaries.get("reference", {}).get("merged_count", 0),
            "molecule": file_summaries.get("molecule", {}).get("merged_count", 0),
        },
        "conflict_counts": {
            "reference": file_summaries.get("reference", {}).get("conflict_count", 0),
            "molecule": file_summaries.get("molecule", {}).get("conflict_count", 0),
        },
        "canonical_counts": {
            "reference": file_summaries.get("reference", {}).get("canonical_count", 0),
            "molecule": file_summaries.get("molecule", {}).get("canonical_count", 0),
        },
        "conflicts": conflicts,
    }


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


def _validate_shard(*, shard_count: int, shard_index: int) -> None:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be >= 0 and < shard_count")


def _select_shard(
    requirements: Iterable[CacheRequirement],
    *,
    shard_count: int,
    shard_index: int,
) -> list[CacheRequirement]:
    return [
        requirement
        for requirement in requirements
        if _stable_shard_index(requirement, shard_count) == shard_index
    ]


def _stable_shard_index(requirement: CacheRequirement, shard_count: int) -> int:
    key = f"{requirement.entity_type}\0{requirement.key}".encode("utf-8")
    return int(sha256(key).hexdigest(), 16) % shard_count


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


def _merge_cache_file(
    canonical_path: Path, shard_paths: Iterable[Path]
) -> dict[str, Any]:
    merged_cache = _load_cache(canonical_path)
    source_by_key = {key: str(canonical_path) for key in merged_cache}
    merged_count = 0
    conflicts: list[dict[str, str]] = []
    for shard_path in shard_paths:
        shard_cache = _load_required_cache(shard_path)
        for key, value in sorted(shard_cache.items()):
            if key in merged_cache:
                if merged_cache[key] == value:
                    continue
                conflicts.append(
                    {
                        "cache_file": canonical_path.name,
                        "key": key,
                        "kept_source": source_by_key[key],
                        "discarded_source": str(shard_path),
                        "kept_sha256": _payload_sha256(merged_cache[key]),
                        "discarded_sha256": _payload_sha256(value),
                    }
                )
                continue
            merged_cache[key] = value
            source_by_key[key] = str(shard_path)
            merged_count += 1
    return {
        "cache": merged_cache,
        "canonical_count": len(merged_cache),
        "merged_count": merged_count,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }


def _load_required_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Required evidence shard cache file is missing: {path}")
    return _load_cache(path)


def _initialize_cache_files(
    *,
    reference_cache: dict[str, Any],
    reference_cache_path: Path,
    molecule_cache: dict[str, Any],
    molecule_cache_path: Path,
) -> None:
    if not reference_cache_path.exists():
        _write_cache(reference_cache, reference_cache_path)
    if not molecule_cache_path.exists():
        _write_cache(molecule_cache, molecule_cache_path)


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _write_cache(cache: dict[str, Any], path: Path) -> None:
    _write_json_atomic(cache, path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


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
