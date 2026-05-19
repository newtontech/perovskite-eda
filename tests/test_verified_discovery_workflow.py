import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _record(record_id: str, doi: str, smiles: str, delta: float, *, title: str = "Verified PSC additive paper") -> dict:
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


def _authenticator():
    from harness.authenticity import MoleculeEvidence, RealDataAuthenticator, ReferenceEvidence

    return RealDataAuthenticator(
        reference_verifier=lambda doi: ReferenceEvidence(
            doi=doi,
            title="Verified PSC additive paper",
            year=2026,
            journal="Journal of Physical Chemistry Letters",
            source="fixture-crossref",
            url=f"https://doi.org/{doi}",
        ) if doi else None,
        molecule_verifier=lambda record: MoleculeEvidence(
            smiles=record["smiles"],
            pubchem_id=record.get("pubchem_id"),
            cas_number=record.get("cas_number"),
            source="fixture-pubchem",
            url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{record.get('pubchem_id')}",
        ),
    )


def _feature_fn(smiles_series):
    return pd.DataFrame({"smiles_length": smiles_series.astype(str).str.len()})


def test_workflow_generates_training_and_candidate_discovery_artifacts(tmp_path):
    from screening.verified_discovery_workflow import VerifiedDiscoveryWorkflow

    raw_df = pd.DataFrame(
        [
            _record("row-001", "10.1021/acs.jpclett.6c00119", "C", 0.25),
            _record("row-002", "10.1021/acs.jpclett.6c00120", "CC", 0.50),
            _record("row-003", "10.1021/acs.jpclett.6c00121", "CCC", 0.75),
            _record("row-004", "10.1021/acs.jpclett.6c00122", "CCCC", 1.00),
            _record("row-005", "", "CCO", 0.40, title="Missing DOI row"),
        ]
    )

    artifacts = VerifiedDiscoveryWorkflow(output_dir=tmp_path, authenticator=_authenticator()).run_from_dataframe(
        raw_df,
        dataset_id="workflow-fixture",
        model=LinearRegression(),
        feature_fn=_feature_fn,
        top_k=2,
        min_verified_rows=4,
    )

    assert artifacts.verified_train_csv.exists()
    assert artifacts.quarantine_csv.exists()
    assert artifacts.candidate_pool_csv.exists()
    assert artifacts.doi_manifest_json.exists()
    assert artifacts.provenance_json.exists()
    assert artifacts.data_audit_report_md.exists()
    assert artifacts.model_metrics_json.exists()
    assert artifacts.model_manifest_json.exists()
    assert artifacts.ranked_candidates_csv.exists()
    assert artifacts.discovery_manifest_json.exists()
    assert artifacts.discovery_audit_report_md.exists()
    assert artifacts.workflow_manifest_json.exists()

    verified = pd.read_csv(artifacts.verified_train_csv)
    quarantine = pd.read_csv(artifacts.quarantine_csv)
    ranked = pd.read_csv(artifacts.ranked_candidates_csv)

    assert verified["record_id"].tolist() == ["row-001", "row-002", "row-003", "row-004"]
    assert quarantine["record_id"].tolist() == ["row-005"]
    assert "row-005" not in set(ranked["record_id"])
    assert ranked["record_id"].tolist() == ["row-004", "row-003"]
    assert ranked["rank"].tolist() == [1, 2]
    assert ranked["verification_status"].tolist() == ["verified", "verified"]

    metrics = json.loads(artifacts.model_metrics_json.read_text(encoding="utf-8"))
    assert metrics["dataset_id"] == "workflow-fixture"
    assert metrics["train_rows"] == 4
    assert metrics["target_column"] == "delta_pce"
    assert metrics["feature_columns"] == ["smiles_length"]
    assert metrics["model_class"] == "LinearRegression"

    manifest = json.loads(artifacts.workflow_manifest_json.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "workflow-fixture"
    assert manifest["verified_rows"] == 4
    assert manifest["quarantine_rows"] == 1
    assert manifest["ranked_candidates"] == 2
    assert manifest["artifact_policy"] == "verified-light-artifacts-in-git"
    assert set(manifest["outputs"]) == {
        "verified_train_csv",
        "quarantine_csv",
        "candidate_pool_csv",
        "doi_manifest_json",
        "provenance_json",
        "data_audit_report_md",
        "model_metrics_json",
        "model_manifest_json",
        "ranked_candidates_csv",
        "discovery_manifest_json",
        "discovery_audit_report_md",
        "workflow_manifest_json",
    }


def test_workflow_refuses_training_when_strict_gate_finds_too_few_verified_rows(tmp_path):
    from screening.verified_discovery_workflow import InsufficientVerifiedDataError, VerifiedDiscoveryWorkflow

    raw_df = pd.DataFrame([
        _record("row-001", "", "C", 0.25, title="Missing DOI row"),
        _record("row-002", "", "CC", 0.50, title="Missing DOI row"),
    ])

    with pytest.raises(InsufficientVerifiedDataError) as exc:
        VerifiedDiscoveryWorkflow(output_dir=tmp_path, authenticator=_authenticator()).run_from_dataframe(
            raw_df,
            dataset_id="no-verified-fixture",
            model=LinearRegression(),
            feature_fn=_feature_fn,
            top_k=1,
            min_verified_rows=2,
        )

    assert "min_verified_rows=2" in str(exc.value)
