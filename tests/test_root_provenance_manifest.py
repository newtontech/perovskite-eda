import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_root_provenance_manifest_indexes_verified_discovery_report_and_si(tmp_path):
    from reporting.root_provenance_manifest import generate_root_provenance_manifest

    discovery_dir = tmp_path / "verified_discovery"
    report_dir = tmp_path / "report_bundle" / "main_text"
    si_dir = tmp_path / "report_bundle" / "si"
    workflow_manifest = {
        "dataset_id": "strict-source-fixture",
        "artifact_policy": "verified-light-artifacts-in-git",
        "strict_verified_training_only": True,
        "verified_candidate_discovery_only": True,
        "outputs": {
            "verified_train_csv": "dataset/verified_train.csv",
            "model_manifest_json": "model/model_manifest.json",
            "ranked_candidates_csv": "discovery/ranked_candidates.csv",
            "discovery_audit_report_md": "discovery/audit_report.md",
        },
    }
    _write(discovery_dir / "workflow_manifest.json", json.dumps(workflow_manifest))
    _write(discovery_dir / "dataset" / "verified_train.csv", "record_id,smiles\nrow-001,C\n")
    _write(discovery_dir / "model" / "model_manifest.json", '{"strict_verified_training_only": true}\n')
    _write(discovery_dir / "discovery" / "ranked_candidates.csv", "rank,candidate_id\n1,cand-001\n")
    _write(discovery_dir / "discovery" / "audit_report.md", "# Audit\nverified-only\n")
    _write(report_dir / "main_text.md", "# Main report\n")
    _write(report_dir / "claim_ledger.json", '[{"evidence_id": "metric:best_model.r2"}]\n')
    _write(report_dir / "review_report.json", '{"review": {"passed": true}}\n')
    _write(si_dir / "supplementary_information.md", "# SI\n")
    _write(si_dir / "audit_summary.json", '{"passed": true}\n')

    manifest = generate_root_provenance_manifest(discovery_dir, report_dir, si_dir)

    output_path = tmp_path / "report_bundle" / "provenance_manifest.json"
    assert output_path.exists()
    assert manifest == json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "strict-source-fixture"
    assert manifest["artifact_policy"] == "verified-light-artifacts-in-git"
    assert manifest["strict_verified_training_only"] is True
    assert manifest["verified_candidate_discovery_only"] is True

    assert {item["id"] for item in manifest["artifacts"]["dataset"]} == {"verified_train_csv"}
    assert {item["id"] for item in manifest["artifacts"]["model"]} == {"model_manifest_json"}
    assert {item["id"] for item in manifest["artifacts"]["discovery"]} == {"ranked_candidates_csv"}
    assert {item["id"] for item in manifest["artifacts"]["audit"]} == {"discovery_audit_report_md", "audit_summary_json"}
    assert {item["id"] for item in manifest["artifacts"]["report"]} == {"main_text_md"}
    assert {item["id"] for item in manifest["artifacts"]["SI"]} == {"supplementary_information_md"}
    assert {item["id"] for item in manifest["artifacts"]["claim"]} == {"claim_ledger_json"}
    assert {item["id"] for item in manifest["artifacts"]["review"]} == {"review_report_json"}

    verified_train = manifest["artifacts"]["dataset"][0]
    assert verified_train["exists"] is True
    assert verified_train["size_bytes"] == len("record_id,smiles\nrow-001,C\n")
    assert verified_train["sha256_16"]


def test_root_provenance_manifest_records_missing_declared_outputs(tmp_path):
    from reporting.root_provenance_manifest import generate_root_provenance_manifest

    discovery_dir = tmp_path / "verified_discovery"
    report_dir = tmp_path / "bundle" / "main"
    si_dir = tmp_path / "bundle" / "si"
    workflow_manifest = {
        "dataset_id": "missing-output-fixture",
        "artifact_policy": "verified-light-artifacts-in-git",
        "strict_verified_training_only": True,
        "verified_candidate_discovery_only": True,
        "outputs": {"model_metrics_json": "model/model_metrics.json"},
    }
    _write(discovery_dir / "workflow_manifest.json", json.dumps(workflow_manifest))
    report_dir.mkdir(parents=True)
    si_dir.mkdir(parents=True)

    manifest = generate_root_provenance_manifest(discovery_dir, report_dir, si_dir)

    missing = manifest["artifacts"]["model"][0]
    assert missing["id"] == "model_metrics_json"
    assert missing["exists"] is False
    assert missing["size_bytes"] is None
    assert missing["sha256_16"] is None
