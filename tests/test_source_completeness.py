import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "title": "Verified additive record A",
                "authors": "A. Example",
                "journal": "Journal of Physical Chemistry Letters",
                "doi": "10.1021/acs.jpclett.6c00119",
                "abstract": "Perovskite additive study.",
                "smiles": "CCO",
                "pubchem_id": "702",
                "cas_number": "64-17-5",
                "molecular_formula": "C2H6O",
                "molecular_weight": 46.07,
                "h_bond_donors": 1,
                "h_bond_acceptors": 1,
                "rotatable_bonds": 0,
                "tpsa": 20.23,
                "log_p": -0.31,
                "jv_reverse_scan_pce_without_modulator": 18.0,
                "jv_reverse_scan_pce": 18.5,
                "delta_pce": 0.5,
            },
            {
                "title": "Verified additive record B",
                "authors": "B. Example",
                "journal": "",
                "doi": "10.1021/acs.jpclett.6c00120",
                "abstract": None,
                "smiles": "CCN",
                "pubchem_id": "8471",
                "cas_number": "",
                "molecular_formula": "C6H15N",
                "molecular_weight": 101.19,
                "h_bond_donors": 0,
                "h_bond_acceptors": 1,
                "rotatable_bonds": 3,
                "tpsa": 3.24,
                "log_p": 1.44,
                "jv_reverse_scan_pce_without_modulator": 17.0,
                "jv_reverse_scan_pce": 17.2,
                "delta_pce": 0.2,
            },
            {
                "title": "",
                "authors": "",
                "journal": "Advanced Energy Materials",
                "doi": "",
                "abstract": "Missing DOI but has chemistry.",
                "smiles": "CCCC",
                "pubchem_id": "7843",
                "cas_number": "106-97-8",
                "molecular_formula": "C4H10",
                "molecular_weight": 58.12,
                "h_bond_donors": 0,
                "h_bond_acceptors": 0,
                "rotatable_bonds": 1,
                "tpsa": 0.0,
                "log_p": 2.89,
                "jv_reverse_scan_pce_without_modulator": None,
                "jv_reverse_scan_pce": 16.1,
                "delta_pce": None,
            },
        ]
    )


def test_source_completeness_uses_real_psc_source_columns_and_writes_json_markdown_csv(tmp_path):
    from data.source_completeness import (
        SOURCE_COMPLETENESS_SCHEMA_VERSION,
        summarize_source_completeness,
        write_source_completeness_artifacts,
    )

    summary = summarize_source_completeness(
        _source_frame(),
        dataset_id="psc-source-fixture",
        source_name="source-columns-smoke",
    )
    artifacts = write_source_completeness_artifacts(summary, tmp_path / "source_completeness")

    assert artifacts.summary_json.exists()
    assert artifacts.table_csv.exists()
    assert artifacts.markdown.exists()

    payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SOURCE_COMPLETENESS_SCHEMA_VERSION
    assert payload["dataset_id"] == "psc-source-fixture"
    assert payload["row_count"] == 3
    assert payload["audit_scope"] == "column_level_missingness_only"
    assert payload["external_verification"] is False
    assert payload["max_rows"] is None
    assert payload["max_rows_is_smoke_only"] is False
    assert payload["audit_population"] == "loaded_source_table"

    groups = {group["id"]: group for group in payload["groups"]}
    assert {"literature_metadata", "chemical_identity", "molecular_descriptors", "jv_core", "target_derivation"}.issubset(groups)

    literature_columns = {column["column"]: column for column in groups["literature_metadata"]["columns"]}
    assert literature_columns["doi"]["non_missing"] == 2
    assert literature_columns["doi"]["missing"] == 1
    assert literature_columns["doi"]["completeness_fraction"] == 0.666667
    assert literature_columns["abstract"]["missing"] == 1

    identity_columns = {column["column"]: column for column in groups["chemical_identity"]["columns"]}
    assert identity_columns["smiles"]["completeness_fraction"] == 1.0
    assert identity_columns["cas_number"]["missing"] == 1

    markdown = artifacts.markdown.read_text(encoding="utf-8")
    assert "Source Completeness Audit" in markdown
    assert "column-level missingness audit" in markdown
    assert "not external verification" in markdown
    assert "| literature_metadata | doi | yes | 2 | 1 | 0.666667 |" in markdown

    table = pd.read_csv(artifacts.table_csv)
    assert {"group_id", "column", "present", "non_missing", "missing", "completeness_fraction"}.issubset(table.columns)
    assert "doi" in table["column"].tolist()


def test_source_completeness_does_not_fabricate_absent_pdf_source_columns(tmp_path):
    from data.source_completeness import format_source_completeness_markdown, summarize_source_completeness

    summary = summarize_source_completeness(
        _source_frame(),
        dataset_id="psc-source-fixture",
        source_name="source-columns-smoke",
    )
    markdown = format_source_completeness_markdown(summary)
    serialized = json.dumps(summary, sort_keys=True) + markdown

    assert "source_pdf_path" not in serialized
    assert "pdf" not in {group["id"] for group in summary["groups"]}


def test_source_completeness_marks_max_rows_subset_as_smoke_only():
    from data.source_completeness import format_source_completeness_markdown, summarize_source_completeness

    summary = summarize_source_completeness(
        _source_frame().head(2),
        dataset_id="psc-source-fixture",
        source_name="source-columns-smoke",
        max_rows=2,
    )

    assert summary["row_count"] == 2
    assert summary["max_rows"] == 2
    assert summary["max_rows_is_smoke_only"] is True
    assert summary["audit_population"] == "max_rows_subset"

    markdown = format_source_completeness_markdown(summary)
    assert "smoke-only subset" in markdown
    assert "not a full-source audit" in markdown
