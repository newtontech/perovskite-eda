"""Normalize local external candidate source tables into candidate-library-v1.

The builder is deliberately offline-only: it accepts already collected source
tables and refuses rows that lack explicit verification sources. It does not
discover, scrape, or synthesize candidates.
"""

from __future__ import annotations

import json
import re
import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from screening.verified_candidate_discovery import (
    CANDIDATE_LIBRARY_CONTRACT_VERSION,
    CANDIDATE_LIBRARY_REQUIRED_COLUMNS,
    validate_candidate_library_contract,
)


DEFAULT_OPTIONAL_COLUMNS = (
    "pubchem_id",
    "cas_number",
    "vendor_name",
    "vendor_catalog_id",
)
OUTPUT_COLUMNS = tuple(dict.fromkeys((*CANDIDATE_LIBRARY_REQUIRED_COLUMNS, *DEFAULT_OPTIONAL_COLUMNS)))
SOURCE_COLUMN_ALIASES = {
    "cas": "cas_number",
    "cas_no": "cas_number",
    "cas_number": "cas_number",
    "vendor": "vendor_name",
    "vendor_name": "vendor_name",
    "catalog_id": "vendor_catalog_id",
    "vendor_catalog_id": "vendor_catalog_id",
    "availability": "availability_status",
    "availability_status": "availability_status",
    "synthesis": "synthesis_status",
    "synthesis_status": "synthesis_status",
    "safety": "safety_status",
    "safety_status": "safety_status",
}
STATUS_DEFAULTS = {
    "availability_status": "not_assessed",
    "synthesis_status": "not_assessed",
    "safety_status": "not_assessed",
}


@dataclass(frozen=True)
class CandidateLibraryArtifacts:
    """Paths and counts emitted by an external candidate source build."""

    output_dir: Path
    candidate_library_csv: Path
    source_summary_json: Path
    provenance_json: Path
    input_count: int
    output_count: int


class CandidateLibraryBuilder:
    """Build a validated candidate-library-v1 CSV from local source records."""

    def __init__(self, *, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        input_path: str | Path,
        *,
        dataset_id: str,
        source_name: str,
    ) -> CandidateLibraryArtifacts:
        """Load a local CSV/XLSX source table and write normalized artifacts."""

        path = Path(input_path)
        df = _load_table(path)
        return self.build_from_dataframe(
            df,
            dataset_id=dataset_id,
            source_name=source_name,
            input_path=path,
        )

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        dataset_id: str,
        source_name: str,
        input_path: str | Path | None = None,
    ) -> CandidateLibraryArtifacts:
        """Normalize a source dataframe and validate it against candidate-library-v1."""

        normalized = _normalize_source_table(df, source_name=source_name)
        validate_candidate_library_contract(normalized)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = CandidateLibraryArtifacts(
            output_dir=self.output_dir,
            candidate_library_csv=self.output_dir / "candidate_library.csv",
            source_summary_json=self.output_dir / "source_summary.json",
            provenance_json=self.output_dir / "provenance.json",
            input_count=len(df),
            output_count=len(normalized),
        )
        normalized.to_csv(artifacts.candidate_library_csv, index=False)
        _write_json(
            _source_summary(dataset_id, source_name, artifacts, normalized),
            artifacts.source_summary_json,
        )
        _write_json(
            _provenance(dataset_id, source_name, artifacts, df, normalized, input_path),
            artifacts.provenance_json,
        )
        return artifacts


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candidate source table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported candidate source format: {path.suffix}")


def _normalize_source_table(df: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    rows = [_normalize_row(row, source_name=source_name) for row in df.to_dict(orient="records")]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _normalize_row(row: dict[str, Any], *, source_name: str) -> dict[str, Any]:
    canonical = _canonicalize_row(row)
    verification_sources = _verification_sources(canonical.get("verification_sources"))
    source_url = _text(canonical.get("source_url")) or _first_source_url(verification_sources)
    output = {
        "candidate_id": _text(canonical.get("candidate_id")) or _candidate_id(canonical, source_name),
        "smiles": _text(canonical.get("smiles")),
        "source_name": _text(canonical.get("source_name")) or source_name,
        "source_url": source_url,
        "availability_status": _status(canonical, "availability_status"),
        "synthesis_status": _status(canonical, "synthesis_status"),
        "safety_status": _status(canonical, "safety_status"),
        "verification_status": _text(canonical.get("verification_status")),
        "verification_sources": json.dumps(verification_sources, ensure_ascii=False, sort_keys=True),
        "pubchem_id": _text(canonical.get("pubchem_id")),
        "cas_number": _text(canonical.get("cas_number")),
        "vendor_name": _text(canonical.get("vendor_name")),
        "vendor_catalog_id": _text(canonical.get("vendor_catalog_id")),
    }
    return output


def _canonicalize_row(row: dict[str, Any]) -> dict[str, Any]:
    canonical = dict(row)
    for source, target in SOURCE_COLUMN_ALIASES.items():
        if source in row and target not in canonical:
            canonical[target] = row[source]
    return canonical


def _candidate_id(row: dict[str, Any], source_name: str) -> str:
    source_slug = _slug(source_name) or "source"
    pubchem_id = _text(row.get("pubchem_id"))
    cas_number = _text(row.get("cas_number"))
    smiles = _text(row.get("smiles"))
    if pubchem_id:
        identity = f"pubchem-{pubchem_id}"
    elif cas_number:
        identity = f"cas-{cas_number}"
    elif smiles:
        identity = f"smiles-{smiles}"
    else:
        identity = "unidentified"
    return f"{source_slug}:{_slug(identity)}"


def _status(row: dict[str, Any], column: str) -> str:
    return _text(row.get(column)) or STATUS_DEFAULTS[column]


def _verification_sources(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [source for source in value if isinstance(source, dict)]
    if not _text(value):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [source for source in parsed if isinstance(source, dict)]


def _first_source_url(sources: list[dict[str, Any]]) -> str:
    for source in sources:
        url = _text(source.get("url"))
        if url:
            return url
    return ""


def _source_summary(
    dataset_id: str,
    source_name: str,
    artifacts: CandidateLibraryArtifacts,
    df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "candidate_library_contract_version": CANDIDATE_LIBRARY_CONTRACT_VERSION,
        "input_rows": artifacts.input_count,
        "output_rows": artifacts.output_count,
        "source_names": dict(Counter(df["source_name"].map(_text))),
        "availability_status": dict(Counter(df["availability_status"].map(_text))),
        "synthesis_status": dict(Counter(df["synthesis_status"].map(_text))),
        "safety_status": dict(Counter(df["safety_status"].map(_text))),
        "verification_source_kinds": dict(Counter(_source_kinds(df))),
        "primary_source_name": source_name,
        "outputs": {
            "candidate_library_csv": artifacts.candidate_library_csv.name,
            "source_summary_json": artifacts.source_summary_json.name,
            "provenance_json": artifacts.provenance_json.name,
        },
    }


def _source_kinds(df: pd.DataFrame) -> list[str]:
    kinds: list[str] = []
    for value in df["verification_sources"]:
        for source in _verification_sources(value):
            kind = _text(source.get("kind"))
            if kind:
                kinds.append(kind)
    return kinds


def _provenance(
    dataset_id: str,
    source_name: str,
    artifacts: CandidateLibraryArtifacts,
    source_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    input_path: str | Path | None,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "builder": "screening.candidate_library_builder.CandidateLibraryBuilder",
        "candidate_library_contract_version": CANDIDATE_LIBRARY_CONTRACT_VERSION,
        "source_name": source_name,
        "input_path": str(input_path) if input_path is not None else None,
        "input_file": _file_facts(Path(input_path)) if input_path is not None else None,
        "input_rows": artifacts.input_count,
        "output_rows": artifacts.output_count,
        "input_columns": list(source_df.columns),
        "output_columns": list(normalized_df.columns),
        "network_access": "not_used",
        "does_not_generate_candidates": True,
        "publication_grade": False,
        "publication_grade_reason": "offline candidate-library normalization does not re-verify candidate identities or availability",
        "verification_status_policy": "explicit_verified_only",
        "validation": {
            "function": "screening.verified_candidate_discovery.validate_candidate_library_contract",
            "status": "passed",
        },
        "outputs": {
            "candidate_library_csv": artifacts.candidate_library_csv.name,
            "source_summary_json": artifacts.source_summary_json.name,
            "provenance_json": artifacts.provenance_json.name,
        },
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_facts(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "sha256": _sha256(path) if exists and path.is_file() else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
