import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _valid_record() -> dict:
    return {
        "record_id": "row-001",
        "doi": "10.1021/acs.jpclett.6c00119",
        "title": "Machine Learning Accelerated Design of Self-Assembled Monolayers for High-Performance Perovskite Solar Cells",
        "year": 2026,
        "journal": "Journal of Physical Chemistry Letters",
        "smiles": "CCO",
        "pubchem_id": "702",
        "cas_number": "64-17-5",
        "jv_reverse_scan_pce_without_modulator": 18.2,
        "jv_reverse_scan_pce": 20.1,
        "delta_pce": 1.9,
    }


def test_verified_record_requires_reference_and_molecule_evidence():
    from harness.authenticity import MoleculeEvidence, RealDataAuthenticator, ReferenceEvidence

    authenticator = RealDataAuthenticator(
        reference_verifier=lambda doi: ReferenceEvidence(
            doi=doi,
            title="Machine Learning Accelerated Design of Self-Assembled Monolayers for High-Performance Perovskite Solar Cells",
            year=2026,
            journal="Journal of Physical Chemistry Letters",
            source="fixture-crossref",
            url="https://doi.org/10.1021/acs.jpclett.6c00119",
        ),
        molecule_verifier=lambda record: MoleculeEvidence(
            smiles=record["smiles"],
            pubchem_id="702",
            cas_number="64-17-5",
            source="fixture-pubchem",
            url="https://pubchem.ncbi.nlm.nih.gov/compound/702",
        ),
    )

    result = authenticator.verify_record(_valid_record())

    assert result.status == "verified"
    assert result.reasons == []
    assert result.normalized["delta_pce"] == 1.9
    assert {source["source"] for source in result.sources} == {"fixture-crossref", "fixture-pubchem"}


def test_title_conflict_quarantines_reference_even_with_valid_doi():
    from harness.authenticity import RealDataAuthenticator, ReferenceEvidence

    authenticator = RealDataAuthenticator(
        reference_verifier=lambda doi: ReferenceEvidence(
            doi=doi,
            title="A different paper title",
            year=2026,
            journal="Journal of Physical Chemistry Letters",
            source="fixture-crossref",
        ),
        molecule_verifier=lambda record: None,
    )
    record = _valid_record()

    result = authenticator.verify_record(record)

    assert result.status == "quarantine"
    assert "reference_title_mismatch" in result.reasons


def test_invalid_smiles_physical_bounds_and_delta_mismatch_are_quarantined():
    from harness.authenticity import MoleculeEvidence, RealDataAuthenticator, ReferenceEvidence

    authenticator = RealDataAuthenticator(
        reference_verifier=lambda doi: ReferenceEvidence(
            doi=doi,
            title=_valid_record()["title"],
            year=2026,
            source="fixture-crossref",
        ),
        molecule_verifier=lambda record: MoleculeEvidence(
            smiles=record["smiles"],
            pubchem_id=record.get("pubchem_id"),
            cas_number=record.get("cas_number"),
            source="fixture-pubchem",
        ),
    )
    record = _valid_record()
    record.update({
        "smiles": "not a smiles !!!",
        "jv_reverse_scan_pce_without_modulator": 18.2,
        "jv_reverse_scan_pce": 48.0,
        "delta_pce": 12.0,
    })

    result = authenticator.verify_record(record)

    assert result.status == "quarantine"
    assert "invalid_smiles" in result.reasons
    assert "treated_pce_out_of_bounds" in result.reasons
    assert "delta_pce_mismatch" in result.reasons


def test_split_records_keeps_verified_training_rows_separate_from_quarantine():
    from harness.authenticity import MoleculeEvidence, RealDataAuthenticator, ReferenceEvidence

    authenticator = RealDataAuthenticator(
        reference_verifier=lambda doi: ReferenceEvidence(
            doi=doi,
            title=_valid_record()["title"],
            year=2026,
            source="fixture-crossref",
        ) if doi else None,
        molecule_verifier=lambda record: MoleculeEvidence(
            smiles=record["smiles"],
            pubchem_id=record.get("pubchem_id"),
            cas_number=record.get("cas_number"),
            source="fixture-pubchem",
        ),
    )
    missing_doi = _valid_record() | {"record_id": "row-002", "doi": ""}

    split = authenticator.split_records([_valid_record(), missing_doi])

    assert [row["record_id"] for row in split.verified] == ["row-001"]
    assert [row["record_id"] for row in split.quarantine] == ["row-002"]
    assert split.quarantine[0]["verification_status"] == "quarantine"
    assert "missing_doi" in split.quarantine[0]["quarantine_reason"]
