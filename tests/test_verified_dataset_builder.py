import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _record(record_id: str, doi: str, *, title: str | None = None) -> dict:
    return {
        "record_id": record_id,
        "doi": doi,
        "title": title or "Verified PSC additive paper",
        "year": 2026,
        "journal": "Journal of Physical Chemistry Letters",
        "smiles": "CCO",
        "pubchem_id": "702",
        "cas_number": "64-17-5",
        "jv_reverse_scan_pce_without_modulator": 18.0,
        "jv_reverse_scan_pce": 20.5,
        "delta_pce": 2.5,
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


def test_builder_writes_verified_training_quarantine_and_manifests(tmp_path):
    from data.verified_dataset_builder import VerifiedDatasetBuilder

    df = pd.DataFrame([
        _record("row-001", "10.1021/acs.jpclett.6c00119"),
        _record("row-002", "", title="Missing DOI row"),
    ])
    artifacts = VerifiedDatasetBuilder(_authenticator(), output_dir=tmp_path).build_from_dataframe(
        df,
        dataset_id="fixture-additive-dataset",
    )

    assert artifacts.verified_count == 1
    assert artifacts.quarantine_count == 1
    assert artifacts.verified_train_csv.exists()
    assert artifacts.quarantine_csv.exists()
    assert artifacts.candidate_pool_csv.exists()
    assert artifacts.doi_manifest_json.exists()
    assert artifacts.provenance_json.exists()
    assert artifacts.audit_report_md.exists()

    verified = pd.read_csv(artifacts.verified_train_csv)
    quarantine = pd.read_csv(artifacts.quarantine_csv)
    candidate_pool = pd.read_csv(artifacts.candidate_pool_csv)

    assert verified["record_id"].tolist() == ["row-001"]
    assert verified["verification_status"].tolist() == ["verified"]
    assert quarantine["record_id"].tolist() == ["row-002"]
    assert "missing_doi" in quarantine.loc[0, "quarantine_reason"]
    assert candidate_pool["record_id"].tolist() == ["row-001"]
    assert "row-002" not in set(candidate_pool["record_id"])

    doi_manifest = json.loads(artifacts.doi_manifest_json.read_text(encoding="utf-8"))
    assert doi_manifest["dataset_id"] == "fixture-additive-dataset"
    assert doi_manifest["references"][0]["doi"] == "10.1021/acs.jpclett.6c00119"
    assert doi_manifest["references"][0]["source"] == "fixture-crossref"

    provenance = json.loads(artifacts.provenance_json.read_text(encoding="utf-8"))
    assert provenance["input_rows"] == 2
    assert provenance["verified_rows"] == 1
    assert provenance["quarantine_rows"] == 1
    assert provenance["artifact_policy"] == "verified-light-artifacts-in-git"
    assert set(provenance["outputs"]) == {
        "verified_train_csv",
        "quarantine_csv",
        "candidate_pool_csv",
        "doi_manifest_json",
        "provenance_json",
        "audit_report_md",
    }

    report = artifacts.audit_report_md.read_text(encoding="utf-8")
    assert "fixture-additive-dataset" in report
    assert "missing_doi" in report


def test_builder_loads_csv_and_computes_delta_when_missing(tmp_path):
    from data.verified_dataset_builder import VerifiedDatasetBuilder

    record = _record("row-003", "10.1021/acs.jpclett.6c00120")
    record.pop("delta_pce")
    input_csv = tmp_path / "raw.csv"
    pd.DataFrame([record]).to_csv(input_csv, index=False)

    artifacts = VerifiedDatasetBuilder(_authenticator(), output_dir=tmp_path / "out").build(
        input_csv,
        dataset_id="csv-fixture",
    )

    verified = pd.read_csv(artifacts.verified_train_csv)
    assert verified["record_id"].tolist() == ["row-003"]
    assert verified.loc[0, "delta_pce"] == 2.5


def test_builder_assigns_stable_record_ids_when_source_has_none(tmp_path):
    from data.verified_dataset_builder import VerifiedDatasetBuilder

    first = _record("row-001", "10.1021/acs.jpclett.6c00119")
    second = _record("row-002", "", title="Missing DOI row")
    for row in (first, second):
        row.pop("record_id")

    artifacts = VerifiedDatasetBuilder(_authenticator(), output_dir=tmp_path).build_from_dataframe(
        pd.DataFrame([first, second]),
        dataset_id="missing-record-id-fixture",
    )

    verified = pd.read_csv(artifacts.verified_train_csv)
    quarantine = pd.read_csv(artifacts.quarantine_csv)
    candidate_pool = pd.read_csv(artifacts.candidate_pool_csv)

    assert verified["record_id"].tolist() == ["source_row_000001"]
    assert quarantine["record_id"].tolist() == ["source_row_000002"]
    assert candidate_pool["record_id"].tolist() == ["source_row_000001"]
