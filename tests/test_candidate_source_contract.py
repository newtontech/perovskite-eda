import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


SEED_SOURCE = PROJECT_ROOT / "candidate_sources" / "psc_passivator_seed_candidates.csv"


def test_tracked_psc_passivator_seed_source_normalizes_to_candidate_library_v1(tmp_path):
    from screening.candidate_library_builder import CandidateLibraryBuilder
    from screening.verified_candidate_discovery import validate_candidate_library_contract

    artifacts = CandidateLibraryBuilder(output_dir=tmp_path / "candidate_library").build(
        SEED_SOURCE,
        dataset_id="psc-passivator-seed-candidates",
        source_name="psc-passivator-seed",
    )

    library = pd.read_csv(artifacts.candidate_library_csv)
    validate_candidate_library_contract(library)

    assert artifacts.input_count >= 5
    assert artifacts.input_count == artifacts.output_count
    assert library["verification_status"].tolist() == ["verified"] * len(library)
    assert not library["source_url"].str.contains("example.com", case=False, na=False).any()
    assert not library["verification_sources"].str.contains("example.com", case=False, na=False).any()

    for row in library.to_dict(orient="records"):
        sources = json.loads(row["verification_sources"])
        molecule_sources = [source for source in sources if source.get("kind") == "molecule"]
        assert molecule_sources, row["candidate_id"]
        assert any(source.get("pubchem_id") for source in molecule_sources), row["candidate_id"]
        assert any("pubchem.ncbi.nlm.nih.gov" in source.get("url", "") for source in molecule_sources), (
            row["candidate_id"]
        )

    summary = json.loads(artifacts.source_summary_json.read_text(encoding="utf-8"))
    assert summary["candidate_library_contract_version"] == "candidate-library-v1"
    assert summary["verification_source_kinds"]["molecule"] == len(library)
