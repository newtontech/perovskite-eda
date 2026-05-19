import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _requirements_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "entity_type": "reference",
                "key": "10.1021/acs.jpclett.6c00119",
                "key_source": "doi",
                "cache_status": "missing",
                "row_count": 2,
                "record_ids": "row-001;row-002",
                "record_ids_json": json.dumps(["row-001", "row-002"]),
            },
            {
                "entity_type": "reference",
                "key": "10.1038/s41586-026-00001-1",
                "key_source": "doi",
                "cache_status": "missing",
                "row_count": 1,
                "record_ids": "row-003",
                "record_ids_json": json.dumps(["row-003"]),
            },
            {
                "entity_type": "reference",
                "key": "10.1126/science.cached",
                "key_source": "doi",
                "cache_status": "positive",
                "row_count": 1,
                "record_ids": "row-004",
                "record_ids_json": json.dumps(["row-004"]),
            },
            {
                "entity_type": "molecule",
                "key": "pubchem:104810",
                "key_source": "pubchem_id",
                "cache_status": "missing",
                "row_count": 2,
                "record_ids": "row-001;row-002",
                "record_ids_json": json.dumps(["row-001", "row-002"]),
            },
            {
                "entity_type": "molecule",
                "key": "smiles:CCO",
                "key_source": "smiles",
                "cache_status": "missing",
                "row_count": 1,
                "record_ids": "row-005",
                "record_ids_json": json.dumps(["row-005"]),
            },
        ]
    ).to_csv(path, index=False)


def _seed_caches(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True)
    (cache_dir / "reference_cache.json").write_text(
        json.dumps(
            {
                "10.1126/science.cached": {
                    "kind": "reference",
                    "source": "fixture-crossref",
                    "doi": "10.1126/science.cached",
                    "title": "Already cached",
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (cache_dir / "molecule_cache.json").write_text(
        json.dumps({"smiles:cached": None}, indent=2) + "\n",
        encoding="utf-8",
    )


def test_collector_resumes_missing_requirements_without_overwriting_existing_cache(tmp_path):
    from data.evidence_cache_collector import collect_evidence_cache
    from harness.authenticity import MoleculeEvidence, ReferenceEvidence

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "evidence_cache"
    _requirements_csv(requirements)
    _seed_caches(cache_dir)
    reference_calls = []
    molecule_calls = []

    def reference_resolver(doi):
        reference_calls.append(doi)
        return ReferenceEvidence(
            doi=doi,
            title=f"Resolved {doi}",
            year=2026,
            journal="Journal of Physical Chemistry Letters",
            source="fixture-crossref",
            url=f"https://doi.org/{doi}",
        )

    def molecule_resolver(record):
        molecule_calls.append(dict(record))
        if record.get("pubchem_id") == "104810":
            return MoleculeEvidence(
                smiles="[Ba+2]",
                pubchem_id="104810",
                source="fixture-pubchem",
                url="https://pubchem.ncbi.nlm.nih.gov/compound/104810",
            )
        return None

    first = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=2,
        reference_resolver=reference_resolver,
        molecule_resolver=molecule_resolver,
    )

    assert first["attempted_count"] == 2
    assert first["remaining_missing_count"] == 2
    assert reference_calls == ["10.1021/acs.jpclett.6c00119", "10.1038/s41586-026-00001-1"]
    assert molecule_calls == []

    second = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=10,
        reference_resolver=reference_resolver,
        molecule_resolver=molecule_resolver,
    )

    assert second["attempted_count"] == 1
    assert second["positive_written_count"] == 1
    assert second["negative_written_count"] == 0
    assert second["unsupported_count"] == 1
    assert second["remaining_missing_count"] == 1
    assert molecule_calls == [{"pubchem_id": "104810"}]

    reference_cache = json.loads((cache_dir / "reference_cache.json").read_text(encoding="utf-8"))
    molecule_cache = json.loads((cache_dir / "molecule_cache.json").read_text(encoding="utf-8"))
    assert reference_cache["10.1126/science.cached"]["title"] == "Already cached"
    assert reference_cache["10.1021/acs.jpclett.6c00119"]["source"] == "fixture-crossref"
    assert reference_cache["10.1038/s41586-026-00001-1"]["source"] == "fixture-crossref"
    assert molecule_cache["pubchem:104810"]["pubchem_id"] == "104810"
    assert "smiles:CCO" not in molecule_cache


def test_collector_dry_run_does_not_write_cache_files(tmp_path):
    from data.evidence_cache_collector import collect_evidence_cache

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "evidence_cache"
    _requirements_csv(requirements)
    _seed_caches(cache_dir)
    before = {
        path.name: path.read_text(encoding="utf-8")
        for path in (cache_dir / "reference_cache.json", cache_dir / "molecule_cache.json")
    }

    summary = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=3,
        dry_run=True,
        reference_resolver=lambda doi: None,
        molecule_resolver=lambda record: None,
    )

    assert summary["dry_run"] is True
    assert summary["attempted_count"] == 0
    assert summary["planned_count"] == 3
    assert summary["unsupported_count"] == 1
    assert {
        path.name: path.read_text(encoding="utf-8")
        for path in (cache_dir / "reference_cache.json", cache_dir / "molecule_cache.json")
    } == before


def test_collector_cli_writes_report_with_injected_resolvers(tmp_path, monkeypatch):
    import run_evidence_cache_collector as runner
    from harness.authenticity import ReferenceEvidence

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "evidence_cache"
    output_json = tmp_path / "collector_report.json"
    _requirements_csv(requirements)
    _seed_caches(cache_dir)

    monkeypatch.setattr(
        runner,
        "CrossrefReferenceVerifier",
        lambda: lambda doi: ReferenceEvidence(doi=doi, title="CLI resolved", source="fixture-crossref"),
    )
    monkeypatch.setattr(runner, "PubChemMoleculeVerifier", lambda: lambda record: None)

    exit_code = runner.main(
        [
            "--requirements-csv",
            str(requirements),
            "--dataset-id",
            "collector-cli",
            "--cache-dir",
            str(cache_dir),
            "--max-requests",
            "1",
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["dataset_id"] == "collector-cli"
    assert report["attempted_count"] == 1
    assert report["entity_type_counts"]["reference"] == 1


def test_collector_retries_transient_errors_without_writing_negative_cache(tmp_path):
    from data.evidence_cache_collector import collect_evidence_cache
    from harness.authenticity import ReferenceEvidence

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "evidence_cache"
    _requirements_csv(requirements)
    _seed_caches(cache_dir)
    attempts = []

    def flaky_reference_resolver(doi):
        attempts.append(doi)
        if len(attempts) == 1:
            raise RuntimeError("temporary crossref failure")
        return ReferenceEvidence(doi=doi, title="Recovered reference", source="fixture-crossref")

    summary = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=1,
        retry_attempts=2,
        reference_resolver=flaky_reference_resolver,
        molecule_resolver=lambda record: None,
    )

    assert attempts == ["10.1021/acs.jpclett.6c00119", "10.1021/acs.jpclett.6c00119"]
    assert summary["attempted_count"] == 1
    assert summary["error_count"] == 0
    reference_cache = json.loads((cache_dir / "reference_cache.json").read_text(encoding="utf-8"))
    assert reference_cache["10.1021/acs.jpclett.6c00119"]["title"] == "Recovered reference"


def test_collector_does_not_write_negative_cache_after_retry_exhaustion(tmp_path):
    from data.evidence_cache_collector import collect_evidence_cache

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "evidence_cache"
    _requirements_csv(requirements)
    _seed_caches(cache_dir)

    def failing_reference_resolver(doi):
        raise RuntimeError("rate limited")

    summary = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=1,
        retry_attempts=2,
        reference_resolver=failing_reference_resolver,
        molecule_resolver=lambda record: None,
    )

    assert summary["attempted_count"] == 1
    assert summary["error_count"] == 1
    reference_cache = json.loads((cache_dir / "reference_cache.json").read_text(encoding="utf-8"))
    assert "10.1021/acs.jpclett.6c00119" not in reference_cache


def test_collector_does_not_write_negative_cache_for_none_by_default(tmp_path):
    from data.evidence_cache_collector import collect_evidence_cache

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "evidence_cache"
    _requirements_csv(requirements)
    _seed_caches(cache_dir)

    summary = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=1,
        reference_resolver=lambda doi: None,
        molecule_resolver=lambda record: None,
    )

    assert summary["attempted_count"] == 1
    assert summary["no_evidence_count"] == 1
    assert summary["negative_written_count"] == 0
    reference_cache = json.loads((cache_dir / "reference_cache.json").read_text(encoding="utf-8"))
    assert "10.1021/acs.jpclett.6c00119" not in reference_cache


def test_collector_never_writes_negative_cache_for_smiles_with_cid_only_resolver(tmp_path):
    from data.evidence_cache_collector import collect_evidence_cache

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "evidence_cache"
    _requirements_csv(requirements)
    _seed_caches(cache_dir)

    summary = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=10,
        entity_type="molecule",
        include_smiles=True,
        write_negative_cache=True,
        reference_resolver=lambda doi: None,
        molecule_resolver=lambda record: None,
    )

    assert summary["attempted_count"] == 2
    assert summary["negative_written_count"] == 1
    assert summary["no_evidence_count"] == 1
    molecule_cache = json.loads((cache_dir / "molecule_cache.json").read_text(encoding="utf-8"))
    assert molecule_cache["pubchem:104810"] is None
    assert "smiles:CCO" not in molecule_cache


def test_collector_emits_progress_snapshots_during_batch(tmp_path):
    from data.evidence_cache_collector import collect_evidence_cache
    from harness.authenticity import ReferenceEvidence

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "evidence_cache"
    _requirements_csv(requirements)
    _seed_caches(cache_dir)
    snapshots = []

    def reference_resolver(doi):
        if doi == "10.1038/s41586-026-00001-1":
            raise RuntimeError("temporary resolver failure")
        return ReferenceEvidence(doi=doi, title="Resolved reference", source="fixture-crossref")

    summary = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=3,
        retry_attempts=1,
        progress_every=1,
        progress_callback=snapshots.append,
        reference_resolver=reference_resolver,
        molecule_resolver=lambda record: None,
    )

    assert summary["attempted_count"] == 3
    assert [snapshot["attempted_count"] for snapshot in snapshots] == [1, 2, 3]
    assert [snapshot["remaining_planned_count"] for snapshot in snapshots] == [2, 1, 0]
    assert [snapshot["remaining_missing_count"] for snapshot in snapshots] == [3, 3, 3]
    assert snapshots[0]["positive_written_count"] == 1
    assert snapshots[1]["error_count"] == 1
    assert snapshots[2]["no_evidence_count"] == 1


def test_collector_cli_prints_progress_and_updates_report(tmp_path, monkeypatch, capsys):
    import run_evidence_cache_collector as runner
    from harness.authenticity import ReferenceEvidence

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "evidence_cache"
    output_json = tmp_path / "collector_report.json"
    _requirements_csv(requirements)
    _seed_caches(cache_dir)

    monkeypatch.setattr(
        runner,
        "CrossrefReferenceVerifier",
        lambda: lambda doi: ReferenceEvidence(doi=doi, title="CLI resolved", source="fixture-crossref"),
    )
    monkeypatch.setattr(runner, "PubChemMoleculeVerifier", lambda: lambda record: None)

    exit_code = runner.main(
        [
            "--requirements-csv",
            str(requirements),
            "--dataset-id",
            "collector-cli-progress",
            "--cache-dir",
            str(cache_dir),
            "--max-requests",
            "2",
            "--output-json",
            str(output_json),
            "--progress-every",
            "1",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "[evidence-cache-collector] progress attempted=1/2" in stdout
    assert "[evidence-cache-collector] progress attempted=2/2" in stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["dataset_id"] == "collector-cli-progress"
    assert report["attempted_count"] == 2
    assert report["remaining_planned_count"] == 0
