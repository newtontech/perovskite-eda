import json
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest


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
                        {"kind": "availability", "source": "vendor_catalog", "vendor_name": "TCI"},
                    ]
                ),
            }
        ]
    )


def _write_source_table(path: Path) -> None:
    pd.DataFrame(
        [
            _raw_record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _raw_record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
            _raw_record("row-003", "10.1021/acs.jpclett.6c00121", "CCC", 0.75),
            _raw_record("row-004", "10.1021/acs.jpclett.6c00122", "CCCC", 1.00),
            _raw_record("row-005", "", "CCO", 0.40, title="Missing DOI row"),
        ]
    ).to_csv(path, index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_contract_verifies_complete_research_package_with_candidate_library(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)

    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="contract-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    result = verify_research_package(package.output_dir, require_candidate_library=True)

    assert result["status"] == "passed"
    assert result["package_dir"] == str(package.output_dir)
    assert result["summary"]["verified_rows"] == 4
    assert result["summary"]["quarantine_rows"] == 1
    assert result["summary"]["candidate_library_rows"] == 1
    assert result["checks"]["required_artifacts"]["status"] == "passed"
    assert result["checks"]["root_provenance"]["status"] == "passed"
    assert "candidate_library/candidate_library.csv" in result["checks"]["required_artifacts"]["paths"]
    assert "source_completeness/source_completeness.json" in result["checks"]["required_artifacts"]["paths"]
    assert "verified_discovery/model/model_validation_metrics.json" in result["checks"]["required_artifacts"]["paths"]
    assert "verified_discovery/model/model_validation_predictions.csv" in result["checks"]["required_artifacts"]["paths"]
    assert "verified_discovery/model/model_scaffold_validation_metrics.json" in result["checks"]["required_artifacts"]["paths"]
    assert "verified_discovery/model/model_scaffold_validation_predictions.csv" in result["checks"]["required_artifacts"]["paths"]
    assert "closed_loop/iteration_000/feedback_template.csv" in result["checks"]["required_artifacts"]["paths"]
    assert result["checks"]["closed_loop_iteration_packet"]["status"] == "passed"
    assert "closure_readiness_report.json" in result["checks"]["required_artifacts"]["paths"]
    assert result["checks"]["closure_readiness"]["status"] == "passed"
    assert result["checks"]["closure_readiness"]["overall_ready"] is False


def test_contract_verifies_relative_package_dir_workflow_outputs(tmp_path, monkeypatch):
    from run_research_package import run_research_package
    from verify_research_package import verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)

    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="relative-package-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    monkeypatch.chdir(tmp_path)

    result = verify_research_package(Path(package.output_dir.name), require_candidate_library=True)

    assert result["status"] == "passed"
    assert result["summary"]["dataset_id"] == "relative-package-fixture"


def test_contract_fails_when_required_candidate_library_is_absent(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    _write_source_table(raw_csv)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="missing-candidate-library-fixture",
        evidence_mode="source-columns",
        min_verified_rows=4,
        top_k=1,
    )

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_candidate_library=True)

    assert "candidate_library_dir" in str(exc.value)


def test_contract_fails_when_core_artifact_is_missing(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="missing-artifact-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    missing = package.verified_discovery_dir / "dataset" / "verified_train.csv"
    missing.unlink()

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_candidate_library=True)

    assert "verified_discovery/dataset/verified_train.csv" in str(exc.value)


def test_contract_fails_when_closed_loop_feedback_contains_measurement(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="closed-loop-measurement-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    feedback_path = package.output_dir / "closed_loop" / "iteration_000" / "feedback_template.csv"
    feedback = pd.read_csv(feedback_path, keep_default_na=False)
    feedback.loc[0, "measurement_date"] = "2026-05-20"
    feedback.to_csv(feedback_path, index=False)

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_candidate_library=True)

    assert "measurement_date" in str(exc.value)


def test_contract_fails_when_validation_metrics_artifact_is_missing(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="missing-validation-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    missing = package.verified_discovery_dir / "model" / "model_validation_metrics.json"
    missing.unlink()

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_candidate_library=True)

    assert "verified_discovery/model/model_validation_metrics.json" in str(exc.value)


def test_contract_fails_when_scaffold_validation_artifact_is_missing(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="missing-scaffold-validation-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    missing = package.verified_discovery_dir / "model" / "model_scaffold_validation_metrics.json"
    missing.unlink()

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_candidate_library=True)

    assert "verified_discovery/model/model_scaffold_validation_metrics.json" in str(exc.value)


def test_contract_fails_when_scaffold_predictions_include_train_group(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="bad-scaffold-predictions-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    metrics_path = package.verified_discovery_dir / "model" / "model_scaffold_validation_metrics.json"
    predictions_path = package.verified_discovery_dir / "model" / "model_scaffold_validation_predictions.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    predictions.loc[0, "chemical_group_key"] = metrics["train_groups"][0]
    predictions.to_csv(predictions_path, index=False)
    root_manifest_path = package.root_provenance_manifest_json
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    for item in root_manifest["artifacts"]["model"]:
        if item["id"] == "model_scaffold_validation_predictions_csv":
            item["size_bytes"] = predictions_path.stat().st_size
            item["sha256_16"] = hashlib.sha256(predictions_path.read_bytes()[:1024 * 1024]).hexdigest()[:16]
            break
    _write_json(root_manifest_path, root_manifest)

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_candidate_library=True)

    assert "scaffold predictions" in str(exc.value)
    assert "train group" in str(exc.value)


def test_contract_fails_strict_closure_readiness_when_not_ready(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="strict-closure-readiness-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(
            package.output_dir,
            require_candidate_library=True,
            require_closure_readiness=True,
        )

    assert "closure readiness required overall_ready=true" in str(exc.value)


def test_contract_rejects_publication_grade_false_without_closure_reason(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="closure-reason-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )
    report_path = package.output_dir / "closure_readiness_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["publication_grade_false_reasons"] = []
    _write_json(report_path, report)

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_candidate_library=True)

    assert "publication_grade_false_reasons must be non-empty" in str(exc.value)


def test_contract_fails_when_root_provenance_record_lies(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="lying-root-provenance-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    root_manifest = json.loads(package.root_provenance_manifest_json.read_text(encoding="utf-8"))
    for item in root_manifest["artifacts"]["dataset"]:
        if item["id"] == "verified_train_csv":
            item["path"] = "NONEXISTENT/verified_train.csv"
            item["exists"] = False
            item["size_bytes"] = 1
            item["sha256_16"] = "deadbeefdeadbeef"
            break
    _write_json(package.root_provenance_manifest_json, root_manifest)

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_candidate_library=True)

    assert "verified_train_csv" in str(exc.value)
    assert "NONEXISTENT/verified_train.csv" in str(exc.value)


def test_contract_fails_when_manifest_output_path_drifts(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="manifest-output-drift-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    package_manifest = json.loads(package.package_manifest_json.read_text(encoding="utf-8"))
    package_manifest["outputs"]["source_completeness_json"] = "source_completeness/does_not_exist.json"
    _write_json(package.package_manifest_json, package_manifest)

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_candidate_library=True)

    assert "outputs.source_completeness_json" in str(exc.value)
    assert "source_completeness/does_not_exist.json" in str(exc.value)


def test_contract_infers_candidate_library_from_declared_candidate_source(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="declared-candidate-source-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    package_manifest = json.loads(package.package_manifest_json.read_text(encoding="utf-8"))
    package_manifest["outputs"]["candidate_library_dir"] = None
    package_manifest["candidate_library_rows"] = None
    _write_json(package.package_manifest_json, package_manifest)
    shutil.rmtree(package.candidate_library_dir)

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir)

    assert "candidate_library" in str(exc.value)


def test_contract_fails_when_required_json_sidecar_is_malformed(tmp_path):
    from run_research_package import run_research_package
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    candidate_csv = tmp_path / "candidate_source.csv"
    _write_source_table(raw_csv)
    _candidate_source().to_csv(candidate_csv, index=False)
    package = run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="malformed-json-fixture",
        evidence_mode="source-columns",
        candidate_source_path=candidate_csv,
        candidate_source_name="fixture-vendor-source",
        min_verified_rows=4,
        top_k=1,
    )

    bad_json = package.report_dir / "main_text" / "claim_ledger.json"
    bad_json.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_candidate_library=True)

    assert "report_bundle/main_text/claim_ledger.json" in str(exc.value)


def test_contract_requires_evidence_cache_manifest_when_cache_is_required(tmp_path, monkeypatch):
    import run_research_package as runner
    from harness.authenticity import RealDataAuthenticator
    from run_verified_discovery import SourceColumnMoleculeVerifier, SourceColumnReferenceVerifier
    from verify_research_package import ResearchPackageContractError, verify_research_package

    raw_csv = tmp_path / "raw_psc.csv"
    _write_source_table(raw_csv)
    cache_dir = tmp_path / "evidence_cache"
    _write_json(cache_dir / "reference_cache.json", {"doi:10.1021/acs.jpclett.6c00119": {"title": "Verified"}})
    _write_json(cache_dir / "molecule_cache.json", {"pubchem:701": {"smiles": "C"}})

    def build_source_column_authenticator(evidence_mode, df, cache_dir):
        return RealDataAuthenticator(
            reference_verifier=SourceColumnReferenceVerifier(df),
            molecule_verifier=SourceColumnMoleculeVerifier(),
        )

    monkeypatch.setattr(runner, "build_authenticator", build_source_column_authenticator)

    package = runner.run_research_package(
        input_path=raw_csv,
        output_dir=tmp_path / "research_package",
        dataset_id="evidence-cache-contract-fixture",
        evidence_mode="external-cached",
        cache_dir=cache_dir,
        input_scope="full-source",
        min_verified_rows=4,
        top_k=1,
    )

    result = verify_research_package(package.output_dir, require_evidence_cache=True)
    assert result["status"] == "passed"
    assert result["checks"]["evidence_cache"]["manifest_path"].endswith(
        "evidence_cache/evidence_cache_manifest.json"
    )
    assert result["checks"]["evidence_cache"]["cache_files"]["reference_cache"]["entry_count"] == 1

    manifest_path = package.output_dir / "evidence_cache" / "evidence_cache_manifest.json"
    evidence_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def refresh_root_manifest_evidence_record() -> None:
        root_manifest = json.loads(package.root_provenance_manifest_json.read_text(encoding="utf-8"))
        for item in root_manifest["artifacts"]["evidence_cache"]:
            if item["id"] == "evidence_cache_manifest_json":
                item["size_bytes"] = manifest_path.stat().st_size
                item["sha256_16"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
                break
        _write_json(package.root_provenance_manifest_json, root_manifest)

    bad_cache_dir_manifest = json.loads(json.dumps(evidence_manifest))
    bad_cache_dir_manifest["cache_dir"] = str(cache_dir / "wrong-cache")
    _write_json(manifest_path, bad_cache_dir_manifest)
    refresh_root_manifest_evidence_record()
    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_evidence_cache=True)
    assert "cache_dir" in str(exc.value)

    bad_file_path_manifest = json.loads(json.dumps(evidence_manifest))
    bad_file_path_manifest["cache_files"]["reference_cache"]["path"] = str(cache_dir / "wrong-reference.json")
    _write_json(manifest_path, bad_file_path_manifest)
    refresh_root_manifest_evidence_record()
    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_evidence_cache=True)
    assert "cache_files.reference_cache.path" in str(exc.value)

    _write_json(manifest_path, evidence_manifest)
    refresh_root_manifest_evidence_record()
    manifest_path.unlink()
    with pytest.raises(ResearchPackageContractError) as exc:
        verify_research_package(package.output_dir, require_evidence_cache=True)

    assert "evidence_cache/evidence_cache_manifest.json" in str(exc.value)
