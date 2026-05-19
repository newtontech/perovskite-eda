import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class LengthModel:
    def predict(self, features):
        return features[:, 0] * 0.25


def _feature_fn(smiles_series):
    return pd.DataFrame({"smiles_length": smiles_series.astype(str).str.len()})


def _verified_pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "row-001",
                "smiles": "CC",
                "doi": "10.1021/acs.jpclett.6c00119",
                "verification_status": "verified",
                "verification_sources": json.dumps([
                    {"kind": "reference", "source": "fixture-crossref", "doi": "10.1021/acs.jpclett.6c00119"}
                ]),
            },
            {
                "record_id": "row-002",
                "smiles": "CCCC",
                "doi": "10.1021/acs.jpclett.6c00120",
                "verification_status": "verified",
                "verification_sources": json.dumps([
                    {"kind": "reference", "source": "fixture-crossref", "doi": "10.1021/acs.jpclett.6c00120"}
                ]),
            },
        ]
    )


def _external_candidate_pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "cand-001",
                "smiles": "CC",
                "pubchem_id": "702",
                "cas_number": "64-17-5",
                "vendor_name": "Sigma Aldrich",
                "vendor_catalog_id": "EtOH",
                "source_name": "pubchem",
                "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/702",
                "availability_status": "commercial",
                "synthesis_status": "commercial",
                "safety_status": "sds_available",
                "verification_status": "verified",
                "verification_sources": json.dumps([
                    {"kind": "molecule", "source": "pubchem", "pubchem_id": "702"},
                    {"kind": "availability", "source": "vendor_catalog", "vendor_name": "Sigma Aldrich"},
                ]),
            },
            {
                "candidate_id": "cand-002",
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
                    {"kind": "molecule", "source": "pubchem", "pubchem_id": "7843"},
                    {"kind": "availability", "source": "vendor_catalog", "vendor_name": "TCI"},
                ]),
            },
        ]
    )


def test_discovery_ranks_verified_pool_and_writes_outputs(tmp_path):
    from screening.verified_candidate_discovery import VerifiedCandidateDiscovery

    artifacts = VerifiedCandidateDiscovery(output_dir=tmp_path).discover(
        _verified_pool(),
        model=LengthModel(),
        feature_fn=_feature_fn,
        top_k=1,
        dataset_id="verified-candidate-fixture",
    )

    assert artifacts.ranked_count == 1
    assert artifacts.ranked_candidates_csv.exists()
    assert artifacts.discovery_manifest_json.exists()
    assert artifacts.audit_report_md.exists()

    ranked = pd.read_csv(artifacts.ranked_candidates_csv)
    assert ranked["record_id"].tolist() == ["row-002"]
    assert ranked["rank"].tolist() == [1]
    assert ranked["verification_status"].tolist() == ["verified"]
    assert ranked["predicted_delta_pce"].tolist() == [1.0]
    assert "fixture-crossref" in ranked.loc[0, "verification_sources"]

    manifest = json.loads(artifacts.discovery_manifest_json.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "verified-candidate-fixture"
    assert manifest["input_rows"] == 2
    assert manifest["ranked_rows"] == 1
    assert manifest["requires_verified_candidates"] is True
    assert manifest["outputs"]["ranked_candidates_csv"] == "ranked_candidates.csv"

    report = artifacts.audit_report_md.read_text(encoding="utf-8")
    assert "verified-only" in report
    assert "row-002" in report


def test_discovery_refuses_quarantined_or_unverified_rows(tmp_path):
    from screening.verified_candidate_discovery import (
        UnverifiedCandidatePoolError,
        VerifiedCandidateDiscovery,
    )

    pool = _verified_pool()
    pool.loc[1, "verification_status"] = "quarantine"

    with pytest.raises(UnverifiedCandidatePoolError) as exc:
        VerifiedCandidateDiscovery(output_dir=tmp_path).discover(
            pool,
            model=LengthModel(),
            feature_fn=_feature_fn,
            dataset_id="bad-pool",
        )

    assert "row-002" in str(exc.value)
    assert "verification_status=verified" in str(exc.value)


def test_candidate_library_contract_refuses_missing_required_source_columns(tmp_path):
    from screening.verified_candidate_discovery import CandidateLibraryContractError, VerifiedCandidateDiscovery

    pool = _external_candidate_pool().drop(columns=["source_url"])

    with pytest.raises(CandidateLibraryContractError) as exc:
        VerifiedCandidateDiscovery(output_dir=tmp_path).discover(
            pool,
            model=LengthModel(),
            feature_fn=_feature_fn,
            dataset_id="missing-source-url",
        )

    assert "source_url" in str(exc.value)


def test_candidate_library_contract_refuses_empty_verification_sources(tmp_path):
    from screening.verified_candidate_discovery import CandidateLibraryContractError, VerifiedCandidateDiscovery

    pool = _external_candidate_pool()
    pool.loc[0, "verification_sources"] = "[]"

    with pytest.raises(CandidateLibraryContractError) as exc:
        VerifiedCandidateDiscovery(output_dir=tmp_path).discover(
            pool,
            model=LengthModel(),
            feature_fn=_feature_fn,
            dataset_id="empty-sources",
        )

    assert "verification_sources" in str(exc.value)
    assert "cand-001" in str(exc.value)


def test_candidate_library_contract_requires_identity_or_molecule_source(tmp_path):
    from screening.verified_candidate_discovery import CandidateLibraryContractError, VerifiedCandidateDiscovery

    pool = _external_candidate_pool()
    pool.loc[0, ["pubchem_id", "cas_number"]] = ""
    pool.loc[0, "verification_sources"] = json.dumps([
        {"kind": "availability", "source": "vendor_catalog"}
    ])

    with pytest.raises(CandidateLibraryContractError) as exc:
        VerifiedCandidateDiscovery(output_dir=tmp_path).discover(
            pool,
            model=LengthModel(),
            feature_fn=_feature_fn,
            dataset_id="missing-identity",
        )

    assert "molecule identity" in str(exc.value)
    assert "cand-001" in str(exc.value)


def test_discovery_ranks_verified_external_candidate_library(tmp_path):
    from screening.verified_candidate_discovery import VerifiedCandidateDiscovery

    artifacts = VerifiedCandidateDiscovery(output_dir=tmp_path).discover(
        _external_candidate_pool(),
        model=LengthModel(),
        feature_fn=_feature_fn,
        top_k=1,
        dataset_id="external-candidates",
    )

    ranked = pd.read_csv(artifacts.ranked_candidates_csv)
    assert ranked["candidate_id"].tolist() == ["cand-002"]
    assert ranked["vendor_name"].tolist() == ["TCI"]
    assert ranked["source_url"].tolist() == ["https://example.com/catalog/B0001"]
    assert ranked["availability_status"].tolist() == ["commercial"]
    assert ranked["synthesis_status"].tolist() == ["commercial"]
    assert ranked["safety_status"].tolist() == ["sds_available"]
    assert "pubchem" in ranked.loc[0, "verification_sources"]

    manifest = json.loads(artifacts.discovery_manifest_json.read_text(encoding="utf-8"))
    assert manifest["candidate_library_contract_version"] == "candidate-library-v1"
    assert manifest["requires_verification_sources"] is True
