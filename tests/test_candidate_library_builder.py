import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "smiles": "CCO",
                "pubchem_id": "702",
                "cas": "64-17-5",
                "vendor": "Sigma Aldrich",
                "vendor_catalog_id": "EtOH",
                "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/702",
                "availability": "commercial",
                "synthesis": "commercial",
                "safety": "sds_available",
                "verification_status": "verified",
                "verification_sources": json.dumps(
                    [
                        {"kind": "molecule", "source": "pubchem", "pubchem_id": "702"},
                        {
                            "kind": "availability",
                            "source": "vendor_catalog",
                            "vendor_name": "Sigma Aldrich",
                            "url": "https://www.sigmaaldrich.com/catalog/product/sial/459844",
                        },
                    ]
                ),
            },
            {
                "smiles": "CCN(CC)CC",
                "pubchem_id": "8471",
                "cas": "121-44-8",
                "vendor": "TCI",
                "vendor_catalog_id": "T0424",
                "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/8471",
                "availability": "commercial",
                "synthesis": "commercial",
                "safety": "sds_available",
                "verification_status": "verified",
                "verification_sources": json.dumps(
                    [
                        {"kind": "molecule", "source": "pubchem", "pubchem_id": "8471"},
                        {
                            "kind": "availability",
                            "source": "vendor_catalog",
                            "vendor_name": "TCI",
                            "url": "https://www.tcichemicals.com/US/en/p/T0424",
                        },
                    ]
                ),
            },
        ]
    )


def test_builder_normalizes_lightweight_source_table_and_writes_provenance(tmp_path):
    from screening.candidate_library_builder import CandidateLibraryBuilder
    from screening.verified_candidate_discovery import validate_candidate_library_contract

    source_csv = tmp_path / "real_vendor_fixture.csv"
    _source_rows().to_csv(source_csv, index=False)

    artifacts = CandidateLibraryBuilder(output_dir=tmp_path / "out").build(
        source_csv,
        dataset_id="vendor-source-fixture",
        source_name="pubchem-vendor-fixture",
    )

    assert artifacts.input_count == 2
    assert artifacts.output_count == 2
    assert artifacts.candidate_library_csv.exists()
    assert artifacts.source_summary_json.exists()
    assert artifacts.provenance_json.exists()

    library = pd.read_csv(artifacts.candidate_library_csv)
    validate_candidate_library_contract(library)

    assert library["candidate_id"].tolist() == [
        "pubchem-vendor-fixture:pubchem-702",
        "pubchem-vendor-fixture:pubchem-8471",
    ]
    assert library["cas_number"].tolist() == ["64-17-5", "121-44-8"]
    assert library["vendor_name"].tolist() == ["Sigma Aldrich", "TCI"]
    assert library["availability_status"].tolist() == ["commercial", "commercial"]
    assert library["synthesis_status"].tolist() == ["commercial", "commercial"]
    assert library["safety_status"].tolist() == ["sds_available", "sds_available"]
    assert library["verification_status"].tolist() == ["verified", "verified"]
    assert "pubchem" in library.loc[0, "verification_sources"]

    summary = json.loads(artifacts.source_summary_json.read_text(encoding="utf-8"))
    assert summary["dataset_id"] == "vendor-source-fixture"
    assert summary["candidate_library_contract_version"] == "candidate-library-v1"
    assert summary["input_rows"] == 2
    assert summary["output_rows"] == 2
    assert summary["source_names"] == {"pubchem-vendor-fixture": 2}
    assert summary["availability_status"] == {"commercial": 2}
    assert summary["verification_source_kinds"] == {"availability": 2, "molecule": 2}

    provenance = json.loads(artifacts.provenance_json.read_text(encoding="utf-8"))
    assert provenance["network_access"] == "not_used"
    assert provenance["does_not_generate_candidates"] is True
    assert provenance["input_file"]["path"].endswith("real_vendor_fixture.csv")
    assert provenance["input_file"]["exists"] is True
    assert provenance["input_file"]["size_bytes"] > 0
    assert len(provenance["input_file"]["sha256"]) == 64
    assert provenance["outputs"]["candidate_library_csv"] == "candidate_library.csv"


def test_builder_rejects_rows_without_real_verification_sources(tmp_path):
    from screening.candidate_library_builder import CandidateLibraryBuilder
    from screening.verified_candidate_discovery import CandidateLibraryContractError

    rows = _source_rows()
    rows.loc[0, "verification_sources"] = "[]"

    with pytest.raises(CandidateLibraryContractError) as exc:
        CandidateLibraryBuilder(output_dir=tmp_path / "out").build_from_dataframe(
            rows,
            dataset_id="bad-source",
            source_name="pubchem-vendor-fixture",
        )

    assert "verification_sources" in str(exc.value)
    assert "pubchem-vendor-fixture:pubchem-702" in str(exc.value)


def test_builder_rejects_missing_explicit_verification_status(tmp_path):
    from screening.candidate_library_builder import CandidateLibraryBuilder
    from screening.verified_candidate_discovery import CandidateLibraryContractError

    rows = _source_rows().drop(columns=["verification_status"])

    with pytest.raises(CandidateLibraryContractError) as exc:
        CandidateLibraryBuilder(output_dir=tmp_path / "out").build_from_dataframe(
            rows,
            dataset_id="missing-status",
            source_name="pubchem-vendor-fixture",
        )

    assert "verification_status" in str(exc.value)
    assert "pubchem-vendor-fixture:pubchem-702" in str(exc.value)


def test_builder_records_explicit_verified_status_policy(tmp_path):
    from screening.candidate_library_builder import CandidateLibraryBuilder

    artifacts = CandidateLibraryBuilder(output_dir=tmp_path / "out").build_from_dataframe(
        _source_rows(),
        dataset_id="explicit-status",
        source_name="pubchem-vendor-fixture",
    )

    provenance = json.loads(artifacts.provenance_json.read_text(encoding="utf-8"))
    assert provenance["verification_status_policy"] == "explicit_verified_only"
