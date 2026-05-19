import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_pubchem_verifier_normalizes_excel_float_cid(monkeypatch):
    from harness import authenticity
    from harness.authenticity import PubChemMoleculeVerifier

    calls = []

    class Response:
        status_code = 200

        def json(self):
            return {
                "PropertyTable": {
                    "Properties": [
                        {"CanonicalSMILES": "[Ba+2]"},
                    ]
                }
            }

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(authenticity.requests, "get", fake_get)

    evidence = PubChemMoleculeVerifier(timeout=3)({
        "pubchem_id": "104810.0",
        "cas_number": "22541-12-4",
    })

    assert evidence is not None
    assert evidence.pubchem_id == "104810"
    assert calls == [
        (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/104810/property/CanonicalSMILES/JSON",
            3,
        )
    ]


def test_cached_reference_verifier_deduplicates_and_persists(tmp_path):
    from harness.authenticity import CachedReferenceVerifier, ReferenceEvidence

    calls = []

    def resolver(doi):
        calls.append(doi)
        return ReferenceEvidence(
            doi=doi.upper(),
            title="Verified title",
            year=2026,
            journal="Journal of Physical Chemistry Letters",
            source="fixture-crossref",
            url=f"https://doi.org/{doi}",
        )

    cache_path = tmp_path / "reference_cache.json"
    cached = CachedReferenceVerifier(resolver, cache_path)

    first = cached(" 10.1021/acs.jpclett.6c00119 ")
    second = cached("10.1021/ACS.JPCLETT.6C00119")

    assert calls == ["10.1021/acs.jpclett.6c00119"]
    assert first == second
    assert first.source == "fixture-crossref"

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert list(payload) == ["10.1021/acs.jpclett.6c00119"]
    assert payload["10.1021/acs.jpclett.6c00119"]["title"] == "Verified title"

    reloaded = CachedReferenceVerifier(lambda doi: None, cache_path)
    assert reloaded("10.1021/acs.jpclett.6c00119").title == "Verified title"


def test_cached_molecule_verifier_keys_by_normalized_pubchem_id(tmp_path):
    from harness.authenticity import CachedMoleculeVerifier, MoleculeEvidence

    calls = []

    def resolver(record):
        calls.append(record["pubchem_id"])
        return MoleculeEvidence(
            smiles="[Ba+2]",
            pubchem_id="104810",
            cas_number=record.get("cas_number"),
            source="fixture-pubchem",
            url="https://pubchem.ncbi.nlm.nih.gov/compound/104810",
        )

    cache_path = tmp_path / "molecule_cache.json"
    cached = CachedMoleculeVerifier(resolver, cache_path)

    first = cached({"pubchem_id": "104810.0", "cas_number": "22541-12-4", "smiles": "[Ba+2]"})
    second = cached({"pubchem_id": "104810", "cas_number": "22541-12-4", "smiles": "[Ba+2]"})

    assert calls == ["104810.0"]
    assert first == second
    assert first.pubchem_id == "104810"

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert list(payload) == ["pubchem:104810"]
    assert payload["pubchem:104810"]["smiles"] == "[Ba+2]"
