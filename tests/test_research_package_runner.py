import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _raw_record(record_id: str, doi: str, smiles: str, delta: float, *, title: str = "Verified PSC additive paper") -> dict:
    return {
        "record_id": record_id,
        "doi": doi,
        "title": title,
        "year": 2026,
        "journal": "Journal of Physical Chemistry Letters",
        "smiles": smiles,
        "pubchem_id": str(700 + int(record_id.split("-")[-1])),
        "cas_number": f"64-17-{record_id[-1]}",
        "jv_reverse_scan_pce_without_modulator": 18.0,
        "jv_reverse_scan_pce": 18.0 + delta,
        "delta_pce": delta,
    }


def _candidate_source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "smiles": "CCCC",
                "pubchem_id": "7843",
                "cas": "106-97-8",
                "vendor": "TCI",
                "vendor_catalog_id": "B0001",
                "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/7843",
                "availability": "commercial",
                "synthesis": "commercial",
                "safety": "sds_available",
                "verification_sources": json.dumps(
                    [
                        {"kind": "molecule", "source": "pubchem", "pubchem_id": "7843"},
                        {"kind": "availability", "source": "vendor_catalog", "vendor_name": "TCI"},
                    ]
                ),
            }
        ]
    )


def test_research_package_runner_generates_all_scientific_artifacts(tmp_path):
    from run_research_package import run_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    output_dir = tmp_path / "research_package"
    pd.DataFrame(
        [
            _raw_record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _raw_record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
            _raw_record("row-003", "10.1021/acs.jpclett.6c00121", "CCC", 0.75),
            _raw_record("row-004", "10.1021/acs.jpclett.6c00122", "CCCC", 1.00),
            _raw_record("row-005", "", "CCO", 0.40, title="Missing DOI row"),
        ]
    ).to_csv(raw_csv, index=False)
    _candidate_source().to_csv(candidate_csv, index=False)

    package = run_research_package(
        input_path=raw_csv,
        output_dir=output_dir,
        dataset_id="package-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
        report_quality_target="top-journal",
    )

    required_paths = [
        package.verified_discovery_dir / "dataset" / "verified_train.csv",
        package.verified_discovery_dir / "dataset" / "quarantine.csv",
        package.verified_discovery_dir / "dataset" / "doi_manifest.json",
        package.verified_discovery_dir / "dataset" / "provenance.json",
        package.verified_discovery_dir / "dataset" / "data_audit_report.md",
        package.verified_discovery_dir / "model" / "model_metrics.json",
        package.verified_discovery_dir / "model" / "model_manifest.json",
        package.verified_discovery_dir / "discovery" / "ranked_candidates.csv",
        package.verified_discovery_dir / "discovery" / "candidate_discovery_manifest.json",
        package.verified_discovery_dir / "discovery" / "candidate_discovery_audit.md",
        package.verified_discovery_dir / "workflow_manifest.json",
        package.candidate_library_dir / "candidate_library.csv",
        package.candidate_library_dir / "source_summary.json",
        package.candidate_library_dir / "provenance.json",
        package.report_dir / "main_text" / "main_text_report.md",
        package.report_dir / "main_text" / "claim_ledger.json",
        package.report_dir / "main_text" / "review_report.json",
        package.report_dir / "main_text" / "run_manifest.json",
        package.report_dir / "si" / "supporting_information.md",
        package.root_provenance_manifest_json,
        package.package_manifest_json,
    ]
    assert all(path.exists() for path in required_paths)

    package_manifest = json.loads(package.package_manifest_json.read_text(encoding="utf-8"))
    assert package_manifest["dataset_id"] == "package-fixture"
    assert package_manifest["verified_rows"] == 4
    assert package_manifest["quarantine_rows"] == 1
    assert package_manifest["candidate_library_rows"] == 1
    assert package_manifest["ranked_candidates"] == 1
    assert package_manifest["publication_grade"] is False
    assert package_manifest["verification_level"] == "source_columns_only"
    assert package_manifest["source_columns_is_smoke_only"] is True
    assert package_manifest["max_rows"] is None
    assert package_manifest["max_rows_is_smoke_only"] is False
    assert package_manifest["candidate_pool_contract_version"] == "candidate-library-v1"
    assert package_manifest["outputs"]["root_provenance_manifest_json"] == "provenance_manifest.json"

    workflow_manifest = json.loads(
        (package.verified_discovery_dir / "workflow_manifest.json").read_text(encoding="utf-8")
    )
    assert workflow_manifest["publication_grade"] is False
    assert workflow_manifest["verification_level"] == "source_columns_only"
    assert workflow_manifest["source_columns_is_smoke_only"] is True

    candidate_provenance = json.loads(
        (package.candidate_library_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert candidate_provenance["network_access"] == "not_used"
    assert candidate_provenance["does_not_generate_candidates"] is True
    assert candidate_provenance["validation"]["status"] == "passed"

    report_text = (package.report_dir / "main_text" / "main_text_report.md").read_text(encoding="utf-8")
    assert "training-only" in report_text
    unsupported_phrases = [
        "substructural fingerprints (Morgan) outperform",
        "incorporating baseline device PCE as a feature helps",
        "SHAP interpretability analysis suggests",
        "SHAP analysis reveals",
        "excellent predictive ability",
        "external validation was performed",
    ]
    for phrase in unsupported_phrases:
        assert phrase not in report_text

    review_report = json.loads(
        (package.report_dir / "main_text" / "review_report.json").read_text(encoding="utf-8")
    )
    assert review_report["claim_audit"]["unsupported_claims"] == []

    run_manifest = json.loads(
        (package.report_dir / "main_text" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["best_model"]["pearson_r"] is None
    assert run_manifest["evidence_context"]["source_columns_is_smoke_only"] is True
    assert run_manifest["evidence_context"]["metric_scope"] == "training_only"

    si_text = (package.report_dir / "si" / "supporting_information.md").read_text(encoding="utf-8")
    assert "source-columns" in si_text
    assert "smoke-only" in si_text
    assert "training-only" in si_text
    assert "Cross-validation was not supplied" in si_text

    root_manifest = json.loads(package.root_provenance_manifest_json.read_text(encoding="utf-8"))
    assert root_manifest["dataset_id"] == "package-fixture"
    assert root_manifest["strict_verified_training_only"] is True
    assert root_manifest["verified_candidate_discovery_only"] is True
    assert "candidate_library:candidate_library_csv" in {
        item["id"] for item in root_manifest["artifacts"]["discovery"]
    }

    ranked = pd.read_csv(package.verified_discovery_dir / "discovery" / "ranked_candidates.csv")
    assert ranked["candidate_id"].tolist() == ["fixture-vendor-source:pubchem-7843"]
    assert "candidate_score" in ranked.columns
