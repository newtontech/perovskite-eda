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
