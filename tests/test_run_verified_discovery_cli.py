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
    assert manifest["evidence_mode"] == "source-columns"
    assert manifest["verification_level"] == "source_columns_only"
    assert manifest["publication_grade"] is False
    assert manifest["source_columns_is_smoke_only"] is True
    assert manifest["input_path"] == str(input_csv)
    assert manifest["max_rows"] is None
    assert "cache_dir" not in manifest
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
    from run_verified_discovery import build_authenticator, default_cache_dir

    cache_dir = default_cache_dir("external-fixture")
    authenticator = build_authenticator("external-cached", pd.DataFrame(), cache_dir)

    assert cache_dir.as_posix().endswith("hybrid_agent_exploration/.cache/verified_discovery/external-fixture/evidence_cache")
    assert authenticator.reference_verifier.cache_path == cache_dir / "reference_cache.json"
    assert authenticator.molecule_verifier.cache_path == cache_dir / "molecule_cache.json"


def test_cli_defaults_external_cached_subset_to_non_publication_grade(tmp_path, monkeypatch):
    import run_verified_discovery as runner
    from harness.authenticity import RealDataAuthenticator
    from run_verified_discovery import SourceColumnMoleculeVerifier, SourceColumnReferenceVerifier

    input_csv = tmp_path / "raw.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        [
            _record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
        ]
    ).to_csv(input_csv, index=False)

    def build_source_column_authenticator(evidence_mode, df, cache_dir):
        return RealDataAuthenticator(
            reference_verifier=SourceColumnReferenceVerifier(df),
            molecule_verifier=SourceColumnMoleculeVerifier(),
        )

    monkeypatch.setattr(runner, "build_authenticator", build_source_column_authenticator)

    exit_code = runner.main(
        [
            "--input",
            str(input_csv),
            "--output-dir",
            str(output_dir),
            "--dataset-id",
            "external-subset-fixture",
            "--evidence-mode",
            "external-cached",
            "--min-verified-rows",
            "2",
            "--top-k",
            "1",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "workflow_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_scope"] == "selected-subset"
    assert manifest["publication_grade"] is False
    assert manifest["publication_grade_reason"] == "input_scope is not full-source"


def test_cli_full_source_external_cached_can_be_publication_grade(tmp_path, monkeypatch):
    import run_verified_discovery as runner
    from harness.authenticity import RealDataAuthenticator
    from run_verified_discovery import SourceColumnMoleculeVerifier, SourceColumnReferenceVerifier

    input_csv = tmp_path / "raw.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        [
            _record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
        ]
    ).to_csv(input_csv, index=False)

    def build_source_column_authenticator(evidence_mode, df, cache_dir):
        return RealDataAuthenticator(
            reference_verifier=SourceColumnReferenceVerifier(df),
            molecule_verifier=SourceColumnMoleculeVerifier(),
        )

    monkeypatch.setattr(runner, "build_authenticator", build_source_column_authenticator)

    exit_code = runner.main(
        [
            "--input",
            str(input_csv),
            "--output-dir",
            str(output_dir),
            "--dataset-id",
            "external-full-source-fixture",
            "--evidence-mode",
            "external-cached",
            "--input-scope",
            "full-source",
            "--min-verified-rows",
            "2",
            "--top-k",
            "1",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "workflow_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_scope"] == "full-source"
    assert manifest["publication_grade"] is True
    assert manifest["publication_grade_reason"] == "external-cached full-source input"


def test_gitignore_excludes_verified_discovery_runtime_outputs():
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")

    assert "hybrid_agent_exploration/.cache/" in gitignore
    assert "hybrid_agent_exploration/results/verified_discovery_runs/" in gitignore
    assert "hybrid_agent_exploration/results/verified_discovery_smoke*/" in gitignore
    assert "**/evidence_cache/" in gitignore


def test_cli_accepts_external_candidate_pool_path(tmp_path):
    from run_verified_discovery import main

    input_csv = tmp_path / "raw.csv"
    candidate_csv = tmp_path / "external_candidates.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        [
            _record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
        ]
    ).to_csv(input_csv, index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "external-001",
                "smiles": "CCCC",
                "pubchem_id": "7843",
                "cas_number": "106-97-8",
                "vendor_name": "TCI",
                "vendor_catalog_id": "B0001",
                "source_name": "vendor_catalog",
                "source_url": "https://example.com/catalog/B0001",
                "availability_status": "commercial",
                "synthesis_status": "commercial",
                "safety_status": "sds_available",
                "verification_status": "verified",
                "verification_sources": json.dumps([
                    {"kind": "molecule", "source": "pubchem", "pubchem_id": "7843"}
                ]),
            }
        ]
    ).to_csv(candidate_csv, index=False)

    exit_code = main(
        [
            "--input",
            str(input_csv),
            "--candidate-pool",
            str(candidate_csv),
            "--output-dir",
            str(output_dir),
            "--dataset-id",
            "cli-external-candidate-fixture",
            "--evidence-mode",
            "source-columns",
            "--min-verified-rows",
            "2",
            "--top-k",
            "1",
        ]
    )

    assert exit_code == 0
    ranked = pd.read_csv(output_dir / "discovery" / "ranked_candidates.csv")
    assert ranked["candidate_id"].tolist() == ["external-001"]
    assert "candidate_score" in ranked.columns
    assert "score_components" in ranked.columns
    discovery_manifest = json.loads(
        (output_dir / "discovery" / "candidate_discovery_manifest.json").read_text(encoding="utf-8")
    )
    assert discovery_manifest["scoring_policy_version"] == "baseline-aware-candidate-scoring-v1"
    assert discovery_manifest["scoring_policy"]["sort_key"] == "candidate_score"
    manifest = json.loads((output_dir / "workflow_manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_pool_path"] == str(candidate_csv)
    assert manifest["candidate_pool_contract_version"] == "candidate-library-v1"
