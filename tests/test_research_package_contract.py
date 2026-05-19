import json
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
