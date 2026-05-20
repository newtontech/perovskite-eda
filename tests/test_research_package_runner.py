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
                "verification_status": "verified",
                "verification_sources": json.dumps(
                    [
                        {"kind": "molecule", "source": "pubchem", "pubchem_id": "7843"},
                        {
                            "kind": "availability",
                            "source": "vendor_catalog",
                            "vendor_name": "TCI",
                            "url": "https://www.tcichemicals.com/US/en/p/B0001",
                        },
                        {
                            "kind": "safety",
                            "source": "vendor_sds",
                            "vendor_name": "TCI",
                            "url": "https://www.tcichemicals.com/US/en/sds/B0001",
                        },
                    ]
                ),
            }
        ]
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_external_lab_feedback(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "iteration_id": "iteration_000",
                "candidate_id": "candidate-001",
                "smiles": "CCCC",
                "experimental_status": "measured",
                "measured_pce_without_modulator": 18.0,
                "measured_pce_with_modulator": 19.1,
                "measured_delta_pce": 1.1,
                "measurement_date": "2026-05-20",
                "protocol_reference": "lab-protocol-v1",
                "lab_or_operator": "operator-a",
                "instrument_or_batch_id": "batch-001",
                "raw_data_reference": "raw/run-001.csv",
                "data_source": "lab-a",
                "notes": "independent lab measurement row",
            }
        ]
    ).to_csv(path, index=False)


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
        package.verified_discovery_dir / "model" / "model_validation_metrics.json",
        package.verified_discovery_dir / "model" / "model_validation_predictions.csv",
        package.verified_discovery_dir / "model" / "model_scaffold_validation_metrics.json",
        package.verified_discovery_dir / "model" / "model_scaffold_validation_predictions.csv",
        package.verified_discovery_dir / "model" / "model_temporal_validation_metrics.json",
        package.verified_discovery_dir / "model" / "model_temporal_validation_predictions.csv",
        package.verified_discovery_dir / "discovery" / "ranked_candidates.csv",
        package.verified_discovery_dir / "discovery" / "candidate_discovery_manifest.json",
        package.verified_discovery_dir / "discovery" / "candidate_discovery_audit.md",
        package.verified_discovery_dir / "workflow_manifest.json",
        package.candidate_library_dir / "candidate_library.csv",
        package.candidate_library_dir / "source_summary.json",
        package.candidate_library_dir / "external_evidence_audit.json",
        package.candidate_library_dir / "external_evidence_audit.md",
        package.candidate_library_dir / "provenance.json",
        package.report_dir / "main_text" / "main_text_report.md",
        package.report_dir / "main_text" / "claim_ledger.json",
        package.report_dir / "main_text" / "review_report.json",
        package.report_dir / "main_text" / "run_manifest.json",
        package.report_dir / "si" / "supporting_information.md",
        package.output_dir / "source_completeness" / "source_completeness.json",
        package.output_dir / "source_completeness" / "source_completeness.csv",
        package.output_dir / "source_completeness" / "source_completeness.md",
        package.output_dir / "closed_loop" / "iteration_000" / "iteration_manifest.json",
        package.output_dir / "closed_loop" / "iteration_000" / "selected_candidates.csv",
        package.output_dir / "closed_loop" / "iteration_000" / "candidate_packet.md",
        package.output_dir / "closed_loop" / "iteration_000" / "feedback_schema.json",
        package.output_dir / "closed_loop" / "iteration_000" / "feedback_template.csv",
        package.output_dir / "closed_loop" / "iteration_000" / "verification_report.json",
        package.output_dir / "closure_readiness_report.json",
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
    assert package_manifest["dataset_publication_grade"] is False
    assert package_manifest["candidate_library_publication_grade"] is False
    assert package_manifest["schema_version"] == "research-package-manifest-v1"
    assert package_manifest["verification_level"] == "source_columns_only"
    assert package_manifest["source_columns_is_smoke_only"] is True
    assert package_manifest["max_rows"] is None
    assert package_manifest["max_rows_is_smoke_only"] is False
    assert package_manifest["candidate_pool_contract_version"] == "candidate-library-v1"
    assert package_manifest["runner"]["name"] == "run_research_package"
    assert package_manifest["runner"]["args"]["dataset_id"] == "package-fixture"
    assert package_manifest["runner"]["args"]["evidence_mode"] == "source-columns"
    assert package_manifest["runner"]["args"]["min_verified_rows"] == 4
    assert package_manifest["runner"]["args"]["top_k"] == 1
    assert package_manifest["inputs"]["source_table"]["path"].endswith("raw_psc.csv")
    assert package_manifest["inputs"]["source_table"]["exists"] is True
    assert package_manifest["inputs"]["source_table"]["size_bytes"] > 0
    assert len(package_manifest["inputs"]["source_table"]["sha256"]) == 64
    assert package_manifest["inputs"]["candidate_source"]["path"].endswith("candidate_source.csv")
    assert package_manifest["inputs"]["candidate_source"]["source_name"] == "fixture-vendor-source"
    assert package_manifest["inputs"]["candidate_source"]["exists"] is True
    assert len(package_manifest["inputs"]["candidate_source"]["sha256"]) == 64
    assert package_manifest["outputs"]["source_completeness_dir"] == "source_completeness"
    assert package_manifest["outputs"]["source_completeness_json"] == "source_completeness/source_completeness.json"
    assert package_manifest["outputs"]["source_completeness_csv"] == "source_completeness/source_completeness.csv"
    assert package_manifest["outputs"]["source_completeness_md"] == "source_completeness/source_completeness.md"
    assert package_manifest["outputs"]["model_validation_metrics_json"] == (
        "verified_discovery/model/model_validation_metrics.json"
    )
    assert package_manifest["outputs"]["model_validation_predictions_csv"] == (
        "verified_discovery/model/model_validation_predictions.csv"
    )
    assert package_manifest["outputs"]["model_scaffold_validation_metrics_json"] == (
        "verified_discovery/model/model_scaffold_validation_metrics.json"
    )
    assert package_manifest["outputs"]["model_scaffold_validation_predictions_csv"] == (
        "verified_discovery/model/model_scaffold_validation_predictions.csv"
    )
    assert package_manifest["outputs"]["model_temporal_validation_metrics_json"] == (
        "verified_discovery/model/model_temporal_validation_metrics.json"
    )
    assert package_manifest["outputs"]["model_temporal_validation_predictions_csv"] == (
        "verified_discovery/model/model_temporal_validation_predictions.csv"
    )
    assert package_manifest["outputs"]["candidate_external_evidence_audit_json"] == (
        "candidate_library/external_evidence_audit.json"
    )
    assert package_manifest["outputs"]["candidate_external_evidence_audit_md"] == (
        "candidate_library/external_evidence_audit.md"
    )
    assert package_manifest["outputs"]["root_provenance_manifest_json"] == "provenance_manifest.json"
    assert package_manifest["outputs"]["closed_loop_iteration_dir"] == "closed_loop/iteration_000"
    assert package_manifest["outputs"]["closed_loop_iteration_manifest_json"] == (
        "closed_loop/iteration_000/iteration_manifest.json"
    )
    assert package_manifest["outputs"]["closed_loop_selected_candidates_csv"] == (
        "closed_loop/iteration_000/selected_candidates.csv"
    )
    assert package_manifest["outputs"]["closed_loop_feedback_template_csv"] == (
        "closed_loop/iteration_000/feedback_template.csv"
    )
    assert package_manifest["outputs"]["closure_readiness_report_json"] == "closure_readiness_report.json"
    closure_readiness = json.loads((package.output_dir / "closure_readiness_report.json").read_text(encoding="utf-8"))
    assert closure_readiness["schema_version"] == "closure-readiness-report-v1"
    assert closure_readiness["dataset_id"] == "package-fixture"
    assert closure_readiness["overall_ready"] is False
    assert closure_readiness["publication_grade"] is False
    assert closure_readiness["publication_grade_false_reasons"]
    assert closure_readiness["gates"]["validation_artifacts"]["passed"] is True
    assert closure_readiness["gates"]["temporal_validation"]["passed"] is False
    assert closure_readiness["gates"]["chemical_generalization_validation"]["passed"] is False
    assert "validation_rows" in closure_readiness["gates"]["chemical_generalization_validation"]["reason"]
    assert closure_readiness["gates"]["closed_loop_packet"]["passed"] is True
    assert closure_readiness["gates"]["external_validation"]["passed"] is False
    assert closure_readiness["gates"]["experimental_feedback"]["passed"] is False

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
    assert candidate_provenance["publication_grade"] is False
    assert candidate_provenance["verification_status_policy"] == "explicit_verified_only"
    assert candidate_provenance["input_file"]["path"].endswith("candidate_source.csv")
    assert candidate_provenance["input_file"]["exists"] is True
    assert len(candidate_provenance["input_file"]["sha256"]) == 64
    assert candidate_provenance["validation"]["status"] == "passed"
    candidate_evidence_audit = json.loads(
        (package.candidate_library_dir / "external_evidence_audit.json").read_text(encoding="utf-8")
    )
    assert candidate_evidence_audit["schema_version"] == "candidate-library-external-evidence-audit-v1"
    assert candidate_evidence_audit["publication_grade"] is False
    assert candidate_evidence_audit["missing_requirements_by_kind"]["literature"] == 1

    report_text = (package.report_dir / "main_text" / "main_text_report.md").read_text(encoding="utf-8")
    assert "internal random holdout" in report_text
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
    assert run_manifest["evidence_context"]["metric_scope"] == "training_and_internal_holdout"
    assert run_manifest["evidence_context"]["internal_validation"] is True
    assert run_manifest["evidence_context"]["external_validation"] is False
    assert run_manifest["evidence_context"]["candidate_library_publication_grade"] is False
    assert run_manifest["source_completeness"]["audit_scope"] == "column_level_missingness_only"
    assert run_manifest["source_completeness"]["external_verification"] is False

    source_completeness = json.loads(
        (package.output_dir / "source_completeness" / "source_completeness.json").read_text(encoding="utf-8")
    )
    assert source_completeness["row_count"] == 5
    assert source_completeness["external_verification"] is False
    assert source_completeness["max_rows"] is None
    assert source_completeness["max_rows_is_smoke_only"] is False

    si_text = (package.report_dir / "si" / "supporting_information.md").read_text(encoding="utf-8")
    assert "source-columns" in si_text
    assert "smoke-only" in si_text
    assert "internal random holdout" in si_text
    assert "independent external-validation" in si_text
    assert "Candidate-library publication-grade flag: `False`" in si_text
    assert "Source Completeness Audit" in si_text
    assert "column-level missingness audit" in si_text
    assert "Cross-validation was not supplied" in si_text

    report_text = (package.report_dir / "main_text" / "main_text_report.md").read_text(encoding="utf-8")
    assert "Source Completeness Audit" in report_text
    assert "not external verification" in report_text

    root_manifest = json.loads(package.root_provenance_manifest_json.read_text(encoding="utf-8"))
    assert root_manifest["dataset_id"] == "package-fixture"
    assert root_manifest["strict_verified_training_only"] is True
    assert root_manifest["verified_candidate_discovery_only"] is True
    assert {item["id"] for item in root_manifest["artifacts"]["package"]} == {"package_manifest_json"}
    assert root_manifest["source_manifests"]["package_manifest_json"]["exists"] is True
    assert root_manifest["source_inputs"]["input_table"]["path"].endswith("raw_psc.csv")
    assert len(root_manifest["source_inputs"]["input_table"]["sha256"]) == 64
    assert root_manifest["source_inputs"]["candidate_source_table"]["path"].endswith("candidate_source.csv")
    assert len(root_manifest["source_inputs"]["candidate_source_table"]["sha256"]) == 64
    assert "model_validation_metrics_json" in {item["id"] for item in root_manifest["artifacts"]["model"]}
    assert "model_validation_predictions_csv" in {item["id"] for item in root_manifest["artifacts"]["model"]}
    assert "model_scaffold_validation_metrics_json" in {item["id"] for item in root_manifest["artifacts"]["model"]}
    assert "model_scaffold_validation_predictions_csv" in {item["id"] for item in root_manifest["artifacts"]["model"]}
    assert "candidate_library:external_evidence_audit_json" in {
        item["id"] for item in root_manifest["artifacts"]["audit"]
    }
    assert "candidate_library:external_evidence_audit_md" in {
        item["id"] for item in root_manifest["artifacts"]["audit"]
    }
    assert {item["id"] for item in root_manifest["artifacts"]["source_completeness"]} == {
        "source_completeness_csv",
        "source_completeness_json",
        "source_completeness_md",
    }
    assert "candidate_library:candidate_library_csv" in {
        item["id"] for item in root_manifest["artifacts"]["discovery"]
    }
    assert {item["id"] for item in root_manifest["artifacts"]["closed_loop"]} == {
        "candidate_packet_md",
        "feedback_schema_json",
        "feedback_template_csv",
        "iteration_manifest_json",
        "selected_candidates_csv",
        "verification_report_json",
    }
    assert {item["id"] for item in root_manifest["artifacts"]["closure_readiness"]} == {
        "closure_readiness_report_json",
    }

    ranked = pd.read_csv(package.verified_discovery_dir / "discovery" / "ranked_candidates.csv")
    assert ranked["candidate_id"].tolist() == ["fixture-vendor-source:pubchem-7843"]
    assert "candidate_score" in ranked.columns
    feedback = pd.read_csv(
        package.output_dir / "closed_loop" / "iteration_000" / "feedback_template.csv",
        keep_default_na=False,
    )
    assert feedback["candidate_id"].tolist() == ranked["candidate_id"].tolist()
    assert set(feedback["experimental_status"]) == {"pending_external_measurement"}
    for column in ["measured_delta_pce", "measured_pce", "measurement_date", "protocol_reference", "notes"]:
        assert feedback[column].tolist() == [""]


def test_research_package_runner_can_ingest_external_lab_feedback_without_making_closure_ready(tmp_path):
    from run_research_package import run_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    feedback_csv = tmp_path / "external_lab_feedback.csv"
    output_dir = tmp_path / "research_package"
    pd.DataFrame(
        [
            _raw_record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _raw_record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
            _raw_record("row-003", "10.1021/acs.jpclett.6c00121", "CCC", 0.75),
            _raw_record("row-004", "10.1021/acs.jpclett.6c00122", "CCCC", 1.00),
        ]
    ).to_csv(raw_csv, index=False)
    candidate_source = _candidate_source()
    candidate_source["candidate_id"] = "candidate-001"
    candidate_source.to_csv(candidate_csv, index=False)
    _write_external_lab_feedback(feedback_csv)

    package = run_research_package(
        input_path=raw_csv,
        output_dir=output_dir,
        dataset_id="external-lab-feedback-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        experimental_feedback_csv=feedback_csv,
        experimental_feedback_data_scope="external_lab",
        min_verified_rows=4,
        top_k=1,
        report_quality_target="top-journal",
    )

    feedback_dir = package.output_dir / "closed_loop" / "iteration_000" / "experimental_feedback"
    required_feedback_paths = [
        feedback_dir / "validated_feedback.csv",
        feedback_dir / "feedback_vs_prediction.csv",
        feedback_dir / "feedback_ingestion_manifest.json",
        feedback_dir / "feedback_validation_report.json",
    ]
    assert all(path.exists() for path in required_feedback_paths)

    package_manifest = json.loads(package.package_manifest_json.read_text(encoding="utf-8"))
    assert package_manifest["runner"]["args"]["experimental_feedback_csv"].endswith("external_lab_feedback.csv")
    assert package_manifest["runner"]["args"]["experimental_feedback_data_scope"] == "external_lab"
    assert package_manifest["outputs"]["experimental_feedback_dir"] == (
        "closed_loop/iteration_000/experimental_feedback"
    )
    assert package_manifest["outputs"]["experimental_feedback_validated_csv"] == (
        "closed_loop/iteration_000/experimental_feedback/validated_feedback.csv"
    )
    assert package_manifest["outputs"]["experimental_feedback_vs_prediction_csv"] == (
        "closed_loop/iteration_000/experimental_feedback/feedback_vs_prediction.csv"
    )
    assert package_manifest["outputs"]["experimental_feedback_ingestion_manifest_json"] == (
        "closed_loop/iteration_000/experimental_feedback/feedback_ingestion_manifest.json"
    )
    assert package_manifest["outputs"]["experimental_feedback_validation_report_json"] == (
        "closed_loop/iteration_000/experimental_feedback/feedback_validation_report.json"
    )

    root_manifest = json.loads(package.root_provenance_manifest_json.read_text(encoding="utf-8"))
    assert {item["id"] for item in root_manifest["artifacts"]["experimental_feedback"]} == {
        "validated_feedback_csv",
        "feedback_vs_prediction_csv",
        "feedback_ingestion_manifest_json",
        "feedback_validation_report_json",
    }

    closure_readiness = json.loads((package.output_dir / "closure_readiness_report.json").read_text(encoding="utf-8"))
    assert closure_readiness["gates"]["experimental_feedback"]["passed"] is True
    assert closure_readiness["gates"]["external_validation"]["passed"] is False
    assert closure_readiness["overall_ready"] is False

    feedback_template = pd.read_csv(
        package.output_dir / "closed_loop" / "iteration_000" / "feedback_template.csv",
        keep_default_na=False,
    )
    assert set(feedback_template["experimental_status"]) == {"pending_external_measurement"}
    for column in ["measured_delta_pce", "measured_pce", "measurement_date", "protocol_reference", "notes"]:
        assert feedback_template[column].tolist() == [""]


def test_external_cached_candidate_source_does_not_overclaim_publication_grade(tmp_path, monkeypatch):
    import run_research_package as runner
    from harness.authenticity import RealDataAuthenticator
    from run_verified_discovery import SourceColumnMoleculeVerifier, SourceColumnReferenceVerifier

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    cache_dir = tmp_path / "evidence_cache"
    pd.DataFrame(
        [
            _raw_record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _raw_record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
            _raw_record("row-003", "10.1021/acs.jpclett.6c00121", "CCC", 0.75),
            _raw_record("row-004", "10.1021/acs.jpclett.6c00122", "CCCC", 1.00),
        ]
    ).to_csv(raw_csv, index=False)
    _candidate_source().to_csv(candidate_csv, index=False)
    _write_json(cache_dir / "reference_cache.json", {"doi:10.1021/acs.jpclett.6c00119": {"title": "Verified"}})
    _write_json(cache_dir / "molecule_cache.json", {"pubchem:7843": {"smiles": "CCCC"}})

    def build_source_column_authenticator(evidence_mode, df, cache_dir):
        return RealDataAuthenticator(
            reference_verifier=SourceColumnReferenceVerifier(df),
            molecule_verifier=SourceColumnMoleculeVerifier(),
        )

    monkeypatch.setattr(runner, "build_authenticator", build_source_column_authenticator)

    package = runner.run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="external-cached-offline-candidates",
        evidence_mode="external-cached",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        cache_dir=cache_dir,
        min_verified_rows=4,
        top_k=1,
    )

    package_manifest = json.loads(package.package_manifest_json.read_text(encoding="utf-8"))
    evidence_manifest_path = package.output_dir / "evidence_cache" / "evidence_cache_manifest.json"
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    assert package_manifest["input_scope"] == "selected-subset"
    assert package_manifest["dataset_publication_grade"] is False
    assert package_manifest["candidate_library_publication_grade"] is False
    assert package_manifest["publication_grade"] is False
    assert package_manifest["publication_grade_reason"] == "input_scope is not full-source"
    assert package_manifest["publication_grade_reasons"] == [
        "input_scope is not full-source",
        "candidate library is offline-normalized",
    ]
    assert package_manifest["candidate_library_publication_grade_reason"] == "candidate library is offline-normalized"
    assert package_manifest["outputs"]["evidence_cache_dir"] == "evidence_cache"
    assert package_manifest["outputs"]["evidence_cache_manifest_json"] == "evidence_cache/evidence_cache_manifest.json"
    assert evidence_manifest["schema_version"] == "evidence-cache-manifest-v1"
    assert evidence_manifest["dataset_id"] == "external-cached-offline-candidates"
    assert evidence_manifest["cache_files"]["reference_cache"]["entry_count"] == 1
    assert evidence_manifest["cache_files"]["molecule_cache"]["entry_count"] == 1
    assert len(evidence_manifest["cache_files"]["reference_cache"]["sha256"]) == 64

    run_manifest = json.loads(
        (package.report_dir / "main_text" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["evidence_context"]["publication_grade"] is False
    assert run_manifest["evidence_context"]["dataset_publication_grade"] is False
    assert run_manifest["evidence_context"]["candidate_library_publication_grade"] is False
    assert run_manifest["evidence_context"]["input_scope"] == "selected-subset"
    assert run_manifest["evidence_context"]["publication_grade_reasons"] == [
        "input_scope is not full-source",
        "candidate library is offline-normalized",
    ]

    si_text = (package.report_dir / "si" / "supporting_information.md").read_text(encoding="utf-8")
    assert "Publication-grade flag: `False`" in si_text
    assert "Candidate-library publication-grade flag: `False`" in si_text

    root_manifest = json.loads(package.root_provenance_manifest_json.read_text(encoding="utf-8"))
    assert "evidence_cache_manifest_json" in {
        item["id"] for item in root_manifest["artifacts"]["evidence_cache"]
    }


def test_external_cached_full_source_without_candidate_source_can_be_publication_grade(tmp_path, monkeypatch):
    import run_research_package as runner
    from harness.authenticity import RealDataAuthenticator
    from run_verified_discovery import SourceColumnMoleculeVerifier, SourceColumnReferenceVerifier

    raw_csv = tmp_path / "raw_psc.csv"
    pd.DataFrame(
        [
            _raw_record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _raw_record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
            _raw_record("row-003", "10.1021/acs.jpclett.6c00121", "CCC", 0.75),
            _raw_record("row-004", "10.1021/acs.jpclett.6c00122", "CCCC", 1.00),
        ]
    ).to_csv(raw_csv, index=False)

    def build_source_column_authenticator(evidence_mode, df, cache_dir):
        return RealDataAuthenticator(
            reference_verifier=SourceColumnReferenceVerifier(df),
            molecule_verifier=SourceColumnMoleculeVerifier(),
        )

    monkeypatch.setattr(runner, "build_authenticator", build_source_column_authenticator)

    package = runner.run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="external-cached-full-source",
        evidence_mode="external-cached",
        input_scope="full-source",
        min_verified_rows=4,
        top_k=1,
    )

    package_manifest = json.loads(package.package_manifest_json.read_text(encoding="utf-8"))
    assert package_manifest["input_scope"] == "full-source"
    assert package_manifest["dataset_publication_grade"] is True
    assert package_manifest["candidate_library_publication_grade"] is True
    assert package_manifest["publication_grade"] is True
    assert package_manifest["publication_grade_reason"] == "all publication-grade gates passed"
    assert package_manifest["publication_grade_reasons"] == []


def test_external_cached_manifest_captures_post_workflow_cache_state(tmp_path, monkeypatch):
    import run_research_package as runner
    from harness.authenticity import MoleculeEvidence, RealDataAuthenticator, ReferenceEvidence

    raw_csv = tmp_path / "raw_psc.csv"
    cache_dir = tmp_path / "evidence_cache"
    _write_json(cache_dir / "reference_cache.json", {})
    _write_json(cache_dir / "molecule_cache.json", {})
    pd.DataFrame(
        [
            _raw_record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _raw_record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
            _raw_record("row-003", "10.1021/acs.jpclett.6c00121", "CCC", 0.75),
            _raw_record("row-004", "10.1021/acs.jpclett.6c00122", "CCCC", 1.00),
        ]
    ).to_csv(raw_csv, index=False)

    def build_mutating_authenticator(evidence_mode, df, cache_dir):
        cache_root = Path(cache_dir)

        def verify_reference(doi: str) -> ReferenceEvidence:
            reference_cache = json.loads((cache_root / "reference_cache.json").read_text(encoding="utf-8"))
            reference_cache[f"doi:{doi}"] = {"title": "Verified"}
            _write_json(cache_root / "reference_cache.json", reference_cache)
            return ReferenceEvidence(doi=doi, title="Verified PSC additive paper", source="test-cache")

        def verify_molecule(record: dict) -> MoleculeEvidence:
            pubchem_id = str(record["pubchem_id"])
            molecule_cache = json.loads((cache_root / "molecule_cache.json").read_text(encoding="utf-8"))
            molecule_cache[f"pubchem:{pubchem_id}"] = {"smiles": record["smiles"]}
            _write_json(cache_root / "molecule_cache.json", molecule_cache)
            return MoleculeEvidence(smiles=record["smiles"], pubchem_id=pubchem_id, source="test-cache")

        return RealDataAuthenticator(reference_verifier=verify_reference, molecule_verifier=verify_molecule)

    monkeypatch.setattr(runner, "build_authenticator", build_mutating_authenticator)

    package = runner.run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="external-cache-mutation-fixture",
        evidence_mode="external-cached",
        cache_dir=cache_dir,
        input_scope="full-source",
        min_verified_rows=4,
        top_k=1,
    )

    evidence_manifest = json.loads(
        (package.output_dir / "evidence_cache" / "evidence_cache_manifest.json").read_text(encoding="utf-8")
    )
    reference_cache = json.loads((cache_dir / "reference_cache.json").read_text(encoding="utf-8"))
    molecule_cache = json.loads((cache_dir / "molecule_cache.json").read_text(encoding="utf-8"))
    assert evidence_manifest["cache_files"]["reference_cache"]["entry_count"] == len(reference_cache)
    assert evidence_manifest["cache_files"]["molecule_cache"]["entry_count"] == len(molecule_cache)
    assert evidence_manifest["cache_files"]["reference_cache"]["sha256"] == runner._sha256(
        cache_dir / "reference_cache.json"
    )
    assert evidence_manifest["cache_files"]["molecule_cache"]["sha256"] == runner._sha256(
        cache_dir / "molecule_cache.json"
    )


def test_research_package_rejects_invalid_input_scope(tmp_path):
    from run_research_package import run_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    pd.DataFrame(
        [
            _raw_record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _raw_record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
        ]
    ).to_csv(raw_csv, index=False)

    try:
        run_research_package(
            input_path=raw_csv,
            output_dir=tmp_path / "research_package",
            dataset_id="invalid-scope",
            evidence_mode="source-columns",
            input_scope="hand-picked",
            min_verified_rows=2,
            top_k=1,
        )
    except ValueError as exc:
        assert "input_scope" in str(exc)
    else:
        raise AssertionError("invalid input_scope should fail")


def test_external_cached_max_rows_does_not_overclaim_publication_grade(tmp_path, monkeypatch):
    import run_research_package as runner
    from harness.authenticity import RealDataAuthenticator
    from run_verified_discovery import SourceColumnMoleculeVerifier, SourceColumnReferenceVerifier

    raw_csv = tmp_path / "raw_psc.csv"
    pd.DataFrame(
        [
            _raw_record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _raw_record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
            _raw_record("row-003", "10.1021/acs.jpclett.6c00121", "CCC", 0.75),
            _raw_record("row-004", "10.1021/acs.jpclett.6c00122", "CCCC", 1.00),
            _raw_record("row-005", "10.1021/acs.jpclett.6c00123", "CCO", 0.40),
        ]
    ).to_csv(raw_csv, index=False)

    def build_source_column_authenticator(evidence_mode, df, cache_dir):
        return RealDataAuthenticator(
            reference_verifier=SourceColumnReferenceVerifier(df),
            molecule_verifier=SourceColumnMoleculeVerifier(),
        )

    monkeypatch.setattr(runner, "build_authenticator", build_source_column_authenticator)

    package = runner.run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="external-cached-smoke",
        evidence_mode="external-cached",
        max_rows=4,
        min_verified_rows=4,
        top_k=1,
    )

    package_manifest = json.loads(package.package_manifest_json.read_text(encoding="utf-8"))
    assert package_manifest["max_rows"] == 4
    assert package_manifest["max_rows_is_smoke_only"] is True
    assert package_manifest["dataset_publication_grade"] is False
    assert package_manifest["candidate_library_publication_grade"] is True
    assert package_manifest["publication_grade"] is False

    run_manifest = json.loads(
        (package.report_dir / "main_text" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["evidence_context"]["max_rows_is_smoke_only"] is True
    assert run_manifest["evidence_context"]["dataset_publication_grade"] is False
    assert run_manifest["evidence_context"]["publication_grade"] is False
    assert run_manifest["source_completeness"]["max_rows"] == 4
    assert run_manifest["source_completeness"]["max_rows_is_smoke_only"] is True
    assert run_manifest["source_completeness"]["audit_population"] == "max_rows_subset"

    report_text = (package.report_dir / "main_text" / "main_text_report.md").read_text(encoding="utf-8")
    assert "replace smoke evidence with external-cached" not in report_text
    assert "rerun the external-cached workflow without `max_rows`" in report_text

    source_completeness = json.loads(
        (package.output_dir / "source_completeness" / "source_completeness.json").read_text(encoding="utf-8")
    )
    assert source_completeness["max_rows"] == 4
    assert source_completeness["max_rows_is_smoke_only"] is True
    assert source_completeness["audit_population"] == "max_rows_subset"

    source_completeness_md = (
        package.output_dir / "source_completeness" / "source_completeness.md"
    ).read_text(encoding="utf-8")
    assert "smoke-only subset" in source_completeness_md
    assert "not a full-source audit" in source_completeness_md
