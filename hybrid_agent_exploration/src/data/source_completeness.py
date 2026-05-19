"""Column-level source completeness audit for PSC research packages.

This module reports missingness in the input table only. It does not verify DOI,
molecule, supplier, or experimental claims against external services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_COMPLETENESS_SCHEMA_VERSION = "source-completeness-v1"
AUDIT_SCOPE = "column_level_missingness_only"

DEFAULT_SOURCE_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "literature_metadata",
        "Literature metadata",
        (
            "title",
            "authors",
            "journal",
            "volume",
            "issue",
            "pages",
            "doi",
            "abstract",
            "publisher",
            "publication_date",
            "funding_info",
            "references_count",
            "is_referenced_by_count",
        ),
    ),
    (
        "chemical_identity",
        "Chemical identity",
        ("cas_number", "pubchem_id", "smiles", "molecular_formula"),
    ),
    (
        "molecular_descriptors",
        "Molecular descriptors",
        ("molecular_weight", "h_bond_donors", "h_bond_acceptors", "rotatable_bonds", "tpsa", "log_p"),
    ),
    (
        "jv_core",
        "JV device metrics",
        (
            "jv_reverse_scan_pce_without_modulator",
            "jv_reverse_scan_j_sc_without_modulator",
            "jv_reverse_scan_v_oc_without_modulator",
            "jv_reverse_scan_ff_without_modulator",
            "jv_reverse_scan_pce",
            "jv_reverse_scan_j_sc",
            "jv_reverse_scan_v_oc",
            "jv_reverse_scan_ff",
            "jv_hysteresis_index_without_modulator",
            "jv_hysteresis_index",
        ),
    ),
    (
        "target_derivation",
        "Target derivation",
        ("delta_pce", "jv_reverse_scan_pce", "jv_reverse_scan_pce_without_modulator"),
    ),
)


@dataclass(frozen=True)
class SourceCompletenessArtifacts:
    """Paths emitted by a source completeness audit."""

    output_dir: Path
    summary_json: Path
    table_csv: Path
    markdown: Path
    summary: dict[str, Any]


def summarize_source_completeness(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    source_name: str,
    max_rows: int | None = None,
    expected_groups: tuple[tuple[str, str, tuple[str, ...]], ...] | None = None,
) -> dict[str, Any]:
    """Summarize source-column missingness for a PSC input table."""

    row_count = int(len(df))
    max_rows_is_smoke_only = max_rows is not None
    groups = [
        _summarize_group(df, row_count=row_count, group_id=group_id, label=label, columns=columns)
        for group_id, label, columns in (expected_groups or DEFAULT_SOURCE_GROUPS)
    ]
    present_columns = sum(group["present_columns"] for group in groups)
    expected_columns = sum(group["expected_columns"] for group in groups)
    available_cells = sum(group["available_cells"] for group in groups)
    total_cells = sum(group["total_cells"] for group in groups)
    overall_fraction = _fraction(available_cells, total_cells)
    missing_expected_columns = [
        {"group_id": group["id"], "column": column["column"]}
        for group in groups
        for column in group["columns"]
        if not column["present"]
    ]
    return {
        "schema_version": SOURCE_COMPLETENESS_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "dataset_id": dataset_id,
        "source_name": source_name,
        "row_count": row_count,
        "max_rows": max_rows,
        "max_rows_is_smoke_only": max_rows_is_smoke_only,
        "audit_population": "max_rows_subset" if max_rows_is_smoke_only else "loaded_source_table",
        "audit_scope": AUDIT_SCOPE,
        "external_verification": False,
        "interpretation": (
            "This is a column-level missingness audit of the supplied table, not external verification "
            "of DOI, molecule, supplier, or device claims."
        ),
        "overall": {
            "expected_columns": expected_columns,
            "present_columns": present_columns,
            "available_cells": available_cells,
            "total_cells": total_cells,
            "completeness_fraction": overall_fraction,
        },
        "missing_expected_columns": missing_expected_columns,
        "groups": groups,
    }


def write_source_completeness_artifacts(summary: dict[str, Any], output_dir: str | Path) -> SourceCompletenessArtifacts:
    """Write JSON, CSV, and Markdown source-completeness artifacts."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_json = root / "source_completeness.json"
    table_csv = root / "source_completeness.csv"
    markdown = root / "source_completeness.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(_flatten_rows(summary)).to_csv(table_csv, index=False)
    markdown.write_text(format_source_completeness_markdown(summary), encoding="utf-8")
    return SourceCompletenessArtifacts(
        output_dir=root,
        summary_json=summary_json,
        table_csv=table_csv,
        markdown=markdown,
        summary=summary,
    )


def format_source_completeness_markdown(summary: dict[str, Any]) -> str:
    """Format a source completeness summary as a compact Markdown table."""

    overall = summary.get("overall", {})
    lines = [
        "# Source Completeness Audit",
        "",
        "This is a column-level missingness audit of the supplied source table; it is not external verification of DOI, molecule, supplier, or device claims.",
        "",
        f"- Dataset: `{summary.get('dataset_id', 'unknown')}`",
        f"- Source: `{summary.get('source_name', 'unknown')}`",
        f"- Rows audited: `{summary.get('row_count', 0)}`",
        f"- Audit population: `{summary.get('audit_population', 'loaded_source_table')}`",
        f"- Overall completeness fraction: `{overall.get('completeness_fraction', 0.0)}`",
        "",
        "| Group | Column | Present | Non-missing | Missing | Completeness |",
        "|-------|--------|---------|-------------|---------|--------------|",
    ]
    if summary.get("max_rows_is_smoke_only"):
        lines.insert(
            4,
            "This audit covers a smoke-only subset selected by `max_rows`; it is not a full-source audit.",
        )
        lines.insert(5, "")
    for row in _flatten_rows(summary):
        present = "yes" if row["present"] else "no"
        lines.append(
            f"| {row['group_id']} | {row['column']} | {present} | {row['non_missing']} | "
            f"{row['missing']} | {row['completeness_fraction']} |"
        )
    lines.append("")
    return "\n".join(lines)


def compact_source_completeness(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a small report-manifest friendly source completeness summary."""

    return {
        "schema_version": summary.get("schema_version"),
        "dataset_id": summary.get("dataset_id"),
        "source_name": summary.get("source_name"),
        "row_count": summary.get("row_count"),
        "max_rows": summary.get("max_rows"),
        "max_rows_is_smoke_only": summary.get("max_rows_is_smoke_only"),
        "audit_population": summary.get("audit_population"),
        "audit_scope": summary.get("audit_scope"),
        "external_verification": summary.get("external_verification"),
        "overall": summary.get("overall", {}),
        "groups": [
            {
                "id": group.get("id"),
                "label": group.get("label"),
                "expected_columns": group.get("expected_columns"),
                "present_columns": group.get("present_columns"),
                "completeness_fraction": group.get("completeness_fraction"),
                "lowest_completeness_columns": _lowest_columns(group.get("columns", [])),
            }
            for group in summary.get("groups", [])
        ],
    }


def _summarize_group(
    df: pd.DataFrame,
    *,
    row_count: int,
    group_id: str,
    label: str,
    columns: tuple[str, ...],
) -> dict[str, Any]:
    column_summaries = [_summarize_column(df, row_count=row_count, column=column) for column in columns]
    present_columns = sum(1 for column in column_summaries if column["present"])
    available_cells = sum(column["non_missing"] for column in column_summaries)
    total_cells = sum(column["total"] for column in column_summaries)
    return {
        "id": group_id,
        "label": label,
        "expected_columns": len(columns),
        "present_columns": present_columns,
        "available_cells": available_cells,
        "total_cells": total_cells,
        "completeness_fraction": _fraction(available_cells, total_cells),
        "columns": column_summaries,
    }


def _summarize_column(df: pd.DataFrame, *, row_count: int, column: str) -> dict[str, Any]:
    if column not in df.columns:
        return {
            "column": column,
            "present": False,
            "total": row_count,
            "non_missing": 0,
            "missing": row_count,
            "completeness_fraction": 0.0,
        }
    values = df[column]
    non_missing = int((~values.map(_is_missing)).sum())
    missing = int(row_count - non_missing)
    return {
        "column": column,
        "present": True,
        "total": row_count,
        "non_missing": non_missing,
        "missing": missing,
        "completeness_fraction": _fraction(non_missing, row_count),
    }


def _flatten_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in summary.get("groups", []):
        for column in group.get("columns", []):
            rows.append(
                {
                    "group_id": group.get("id"),
                    "group_label": group.get("label"),
                    "column": column.get("column"),
                    "present": column.get("present"),
                    "total": column.get("total"),
                    "non_missing": column.get("non_missing"),
                    "missing": column.get("missing"),
                    "completeness_fraction": column.get("completeness_fraction"),
                }
            )
    return rows


def _lowest_columns(columns: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "column": column.get("column"),
            "present": column.get("present"),
            "missing": column.get("missing"),
            "completeness_fraction": column.get("completeness_fraction"),
        }
        for column in sorted(columns, key=lambda item: (item.get("completeness_fraction", 0.0), item.get("column", "")))[:limit]
    ]


def _is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null"}
    return False


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
