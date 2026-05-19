import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _record(record_id: str, doi: str, smiles: str, delta: float, *, title: str = "Verified source title") -> dict:
    return {
        "record_id": record_id,
        "doi": doi,
        "title": title,
        "publication_date": 2026,
        "journal": "Journal of Physical Chemistry Letters",
        "smiles": smiles,
        "pubchem_id": str(900 + int(record_id.split("-")[-1])),
        "cas_number": f"64-17-{record_id[-1]}",
        "jv_reverse_scan_pce_without_modulator": 18.0,
        "jv_reverse_scan_pce": 18.0 + delta,
    }


def test_cli_runs_source_column_verified_discovery_smoke(tmp_path):
    from run_verified_discovery import main

    input_csv = tmp_path / "raw.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        [
            _record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
            _record("row-003", "", "CCC", 0.75, title="Missing DOI row"),
        ]
    ).to_csv(input_csv, index=False)

    exit_code = main(
        [
            "--input",
            str(input_csv),
            "--output-dir",
            str(output_dir),
            "--dataset-id",
            "cli-source-fixture",
            "--evidence-mode",
            "source-columns",
            "--min-verified-rows",
            "2",
            "--top-k",
            "1",
        ]
    )

    assert exit_code == 0
    workflow_manifest = output_dir / "workflow_manifest.json"
    assert workflow_manifest.exists()
    manifest = json.loads(workflow_manifest.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "cli-source-fixture"
    assert manifest["verified_rows"] == 2
    assert manifest["quarantine_rows"] == 1
    assert manifest["ranked_candidates"] == 1

    verified = pd.read_csv(output_dir / "dataset" / "verified_train.csv")
    quarantine = pd.read_csv(output_dir / "dataset" / "quarantine.csv")
    ranked = pd.read_csv(output_dir / "discovery" / "ranked_candidates.csv")

    assert verified["record_id"].tolist() == ["row-001", "row-002"]
    assert quarantine["record_id"].tolist() == ["row-003"]
    assert ranked["record_id"].tolist() == ["row-002"]
    assert "source-columns" in verified.loc[0, "verification_sources"]


def test_cli_external_cached_mode_initializes_cache_paths(tmp_path):
    from run_verified_discovery import build_authenticator

    cache_dir = tmp_path / "cache"
    authenticator = build_authenticator("external-cached", pd.DataFrame(), cache_dir)

    assert authenticator.reference_verifier.cache_path == cache_dir / "reference_cache.json"
    assert authenticator.molecule_verifier.cache_path == cache_dir / "molecule_cache.json"
