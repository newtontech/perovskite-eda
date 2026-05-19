"""Build strict verified datasets and quarantine artifacts.

This module is intentionally separate from the exploratory cleaner: only rows
that pass explicit literature and molecule authenticity checks enter the
default training CSV. Everything else is preserved in quarantine with reasons.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from harness.authenticity import (
    CrossrefReferenceVerifier,
    PubChemMoleculeVerifier,
    RealDataAuthenticator,
)


ARTIFACT_POLICY = "verified-light-artifacts-in-git"
DEFAULT_CANDIDATE_COLUMNS = (
    "record_id",
    "smiles",
    "pubchem_id",
    "cas_number",
    "molecular_formula",
    "molecular_weight",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "tpsa",
    "log_p",
    "doi",
    "title",
    "journal",
    "year",
    "delta_pce",
    "verification_status",
    "verification_sources",
)


@dataclass(frozen=True)
class VerifiedDatasetArtifacts:
    """Paths and counts emitted by a verified dataset build."""

    output_dir: Path
    verified_train_csv: Path
    quarantine_csv: Path
    candidate_pool_csv: Path
    doi_manifest_json: Path
    provenance_json: Path
    audit_report_md: Path
    input_count: int
    verified_count: int
    quarantine_count: int


class VerifiedDatasetBuilder:
    """Create verified training, quarantine, provenance, and manifest files."""

    def __init__(
        self,
        authenticator: RealDataAuthenticator | None = None,
        *,
        output_dir: str | Path,
        candidate_columns: Iterable[str] = DEFAULT_CANDIDATE_COLUMNS,
    ) -> None:
        self.authenticator = authenticator or RealDataAuthenticator(
            reference_verifier=CrossrefReferenceVerifier(),
            molecule_verifier=PubChemMoleculeVerifier(),
        )
        self.output_dir = Path(output_dir)
        self.candidate_columns = tuple(candidate_columns)

    def build(self, input_path: str | Path, *, dataset_id: str | None = None) -> VerifiedDatasetArtifacts:
        """Load a CSV/XLSX table and write all strict verification artifacts."""

        path = Path(input_path)
        df = _load_table(path)
        return self.build_from_dataframe(df, dataset_id=dataset_id or path.stem)

    def build_from_dataframe(self, df: pd.DataFrame, *, dataset_id: str) -> VerifiedDatasetArtifacts:
        """Verify a dataframe and write the complete artifact bundle."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = df.to_dict(orient="records")
        split = self.authenticator.split_records(records)

        verified_rows = [_serialize_row(row) for row in split.verified]
        quarantine_rows = [_serialize_row(row) for row in split.quarantine]
        candidate_rows = _candidate_pool(verified_rows, self.candidate_columns)

        artifacts = VerifiedDatasetArtifacts(
            output_dir=self.output_dir,
            verified_train_csv=self.output_dir / "verified_train.csv",
            quarantine_csv=self.output_dir / "quarantine.csv",
            candidate_pool_csv=self.output_dir / "candidate_pool.csv",
            doi_manifest_json=self.output_dir / "doi_manifest.json",
            provenance_json=self.output_dir / "provenance.json",
            audit_report_md=self.output_dir / "data_audit_report.md",
            input_count=len(records),
            verified_count=len(verified_rows),
            quarantine_count=len(quarantine_rows),
        )

        _write_csv(verified_rows, artifacts.verified_train_csv, _csv_columns(df, verified_rows))
        _write_csv(quarantine_rows, artifacts.quarantine_csv, _csv_columns(df, quarantine_rows))
        _write_csv(candidate_rows, artifacts.candidate_pool_csv, list(self.candidate_columns))
        _write_json(_doi_manifest(dataset_id, split.verified), artifacts.doi_manifest_json)
        _write_json(_provenance(dataset_id, artifacts, df), artifacts.provenance_json)
        artifacts.audit_report_md.write_text(
            _audit_report(dataset_id, artifacts, split.quarantine),
            encoding="utf-8",
        )
        return artifacts


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input table format: {path.suffix}")


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["verification_sources"] = json.dumps(
        output.get("verification_sources", []),
        ensure_ascii=False,
        sort_keys=True,
    )
    return output


def _csv_columns(df: pd.DataFrame, rows: list[dict[str, Any]]) -> list[str]:
    columns = list(df.columns)
    for extra in ("verification_status", "quarantine_reason", "verification_sources"):
        if extra not in columns:
            columns.append(extra)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _candidate_pool(rows: list[dict[str, Any]], candidate_columns: Iterable[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    columns = tuple(candidate_columns)
    for row in rows:
        key = (_text(row.get("smiles")), _text(row.get("pubchem_id")), _text(row.get("cas_number")))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({column: row.get(column) for column in columns if column in row})
    return candidates


def _doi_manifest(dataset_id: str, verified_rows: list[dict[str, Any]]) -> dict[str, Any]:
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in verified_rows:
        doi = _text(row.get("doi"))
        if not doi or doi.lower() in seen:
            continue
        seen.add(doi.lower())
        reference_source = _first_source(row, "reference")
        references.append(
            {
                "doi": doi,
                "title": _text(row.get("title")),
                "year": _maybe_int(row.get("year")),
                "journal": _text(row.get("journal")) or None,
                "record_ids": [row.get("record_id")],
                "source": reference_source.get("source"),
                "url": reference_source.get("url"),
            }
        )
    return {
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "reference_count": len(references),
        "references": references,
    }


def _first_source(row: dict[str, Any], kind: str) -> dict[str, Any]:
    for source in row.get("verification_sources", []):
        if source.get("kind") == kind:
            return source
    return {}


def _provenance(dataset_id: str, artifacts: VerifiedDatasetArtifacts, df: pd.DataFrame) -> dict[str, Any]:
    outputs = {
        "verified_train_csv": artifacts.verified_train_csv.name,
        "quarantine_csv": artifacts.quarantine_csv.name,
        "candidate_pool_csv": artifacts.candidate_pool_csv.name,
        "doi_manifest_json": artifacts.doi_manifest_json.name,
        "provenance_json": artifacts.provenance_json.name,
        "audit_report_md": artifacts.audit_report_md.name,
    }
    return {
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "artifact_policy": ARTIFACT_POLICY,
        "input_rows": artifacts.input_count,
        "verified_rows": artifacts.verified_count,
        "quarantine_rows": artifacts.quarantine_count,
        "input_columns": list(df.columns),
        "outputs": outputs,
    }


def _audit_report(
    dataset_id: str,
    artifacts: VerifiedDatasetArtifacts,
    quarantine_rows: list[dict[str, Any]],
) -> str:
    reasons = Counter()
    for row in quarantine_rows:
        for reason in _text(row.get("quarantine_reason")).split(";"):
            if reason:
                reasons[reason] += 1

    lines = [
        f"# Data Authenticity Audit: {dataset_id}",
        "",
        f"- Artifact policy: `{ARTIFACT_POLICY}`",
        f"- Input rows: {artifacts.input_count}",
        f"- Verified training rows: {artifacts.verified_count}",
        f"- Quarantined rows: {artifacts.quarantine_count}",
        "",
        "## Quarantine Reasons",
        "",
    ]
    if reasons:
        lines.extend(f"- {reason}: {count}" for reason, count in sorted(reasons.items()))
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{artifacts.verified_train_csv.name}`",
            f"- `{artifacts.quarantine_csv.name}`",
            f"- `{artifacts.candidate_pool_csv.name}`",
            f"- `{artifacts.doi_manifest_json.name}`",
            f"- `{artifacts.provenance_json.name}`",
            f"- `{artifacts.audit_report_md.name}`",
            "",
        ]
    )
    return "\n".join(lines)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _maybe_int(value: Any) -> int | None:
    text = _text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
