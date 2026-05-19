import json
import sys
from pathlib import Path

import pandas as pd
import pytest


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


def test_collector_resumes_missing_requirements_without_overwriting_existing_cache(
    tmp_path,
):
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
    assert reference_calls == [
        "10.1021/acs.jpclett.6c00119",
        "10.1038/s41586-026-00001-1",
    ]
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

    reference_cache = json.loads(
        (cache_dir / "reference_cache.json").read_text(encoding="utf-8")
    )
    molecule_cache = json.loads(
        (cache_dir / "molecule_cache.json").read_text(encoding="utf-8")
    )
    assert reference_cache["10.1126/science.cached"]["title"] == "Already cached"
    assert (
        reference_cache["10.1021/acs.jpclett.6c00119"]["source"] == "fixture-crossref"
    )
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
        for path in (
            cache_dir / "reference_cache.json",
            cache_dir / "molecule_cache.json",
        )
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
        for path in (
            cache_dir / "reference_cache.json",
            cache_dir / "molecule_cache.json",
        )
    } == before


def test_collector_actual_run_initializes_empty_cache_files_for_shard_merge(tmp_path):
    from data.evidence_cache_collector import collect_evidence_cache

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "reference-only-shard"
    _requirements_csv(requirements)

    summary = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=1,
        entity_type="reference",
        reference_resolver=lambda doi: None,
        molecule_resolver=lambda record: None,
    )

    assert summary["attempted_count"] == 1
    assert (cache_dir / "reference_cache.json").is_file()
    assert (cache_dir / "molecule_cache.json").is_file()
    assert json.loads((cache_dir / "reference_cache.json").read_text()) == {}
    assert json.loads((cache_dir / "molecule_cache.json").read_text()) == {}


def test_collector_shard_plans_are_non_overlapping(tmp_path):
    from data.evidence_cache_collector import collect_evidence_cache

    requirements = tmp_path / "requirements.csv"
    shard_zero_cache = tmp_path / "shard-0"
    shard_one_cache = tmp_path / "shard-1"
    _requirements_csv(requirements)

    shard_zero = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=shard_zero_cache,
        max_requests=10,
        dry_run=True,
        shard_count=2,
        shard_index=0,
        include_smiles=True,
        reference_resolver=lambda doi: None,
        molecule_resolver=lambda record: None,
    )
    shard_one = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=shard_one_cache,
        max_requests=10,
        dry_run=True,
        shard_count=2,
        shard_index=1,
        include_smiles=True,
        reference_resolver=lambda doi: None,
        molecule_resolver=lambda record: None,
    )

    shard_zero_keys = {item["key"] for item in shard_zero["planned"]}
    shard_one_keys = {item["key"] for item in shard_one["planned"]}
    assert shard_zero["shard"] == {"count": 2, "index": 0}
    assert shard_one["shard"] == {"count": 2, "index": 1}
    assert shard_zero_keys
    assert shard_one_keys
    assert shard_zero_keys.isdisjoint(shard_one_keys)
    assert shard_zero_keys | shard_one_keys == {
        "10.1021/acs.jpclett.6c00119",
        "10.1038/s41586-026-00001-1",
        "pubchem:104810",
        "smiles:CCO",
    }
    assert not (shard_zero_cache / "reference_cache.json").exists()
    assert not (shard_one_cache / "reference_cache.json").exists()


def test_collector_shard_resume_keeps_stable_ownership(tmp_path):
    from data.evidence_cache_collector import collect_evidence_cache
    from harness.authenticity import MoleculeEvidence, ReferenceEvidence

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "shard-cache"
    _requirements_csv(requirements)

    initial_plans = [
        collect_evidence_cache(
            requirements_csv=requirements,
            cache_dir=cache_dir,
            max_requests=10,
            dry_run=True,
            shard_count=2,
            shard_index=index,
            include_smiles=True,
            reference_resolver=lambda doi: None,
            molecule_resolver=lambda record: None,
        )["planned"]
        for index in (0, 1)
    ]
    shard_index, initial_plan = next(
        (index, plan) for index, plan in enumerate(initial_plans) if len(plan) >= 2
    )
    initial_keys = {item["key"] for item in initial_plan}

    first_batch = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=1,
        shard_count=2,
        shard_index=shard_index,
        include_smiles=True,
        reference_resolver=lambda doi: ReferenceEvidence(
            doi=doi, title="Resolved reference", source="fixture-crossref"
        ),
        molecule_resolver=lambda record: MoleculeEvidence(
            smiles=record.get("smiles", "[Ba+2]"),
            pubchem_id=record.get("pubchem_id"),
            source="fixture-pubchem",
        ),
    )

    resumed = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=10,
        dry_run=True,
        shard_count=2,
        shard_index=shard_index,
        include_smiles=True,
        reference_resolver=lambda doi: None,
        molecule_resolver=lambda record: None,
    )

    processed_keys = {item["key"] for item in first_batch["processed"]}
    resumed_keys = {item["key"] for item in resumed["planned"]}
    assert resumed_keys == initial_keys - processed_keys


def test_merge_evidence_cache_shards_preserves_canonical_and_adds_shards(tmp_path):
    from data.evidence_cache_collector import merge_evidence_cache_shards

    canonical = tmp_path / "canonical"
    shard_a = tmp_path / "shard-a"
    shard_b = tmp_path / "shard-b"
    canonical.mkdir()
    shard_a.mkdir()
    shard_b.mkdir()
    (canonical / "reference_cache.json").write_text(
        json.dumps(
            {
                "10.existing/canonical": {"title": "canonical existing"},
                "10.conflict/ref": {"title": "canonical wins"},
            }
        ),
        encoding="utf-8",
    )
    (canonical / "molecule_cache.json").write_text(
        json.dumps({"pubchem:existing": {"source": "canonical"}}),
        encoding="utf-8",
    )
    (shard_a / "reference_cache.json").write_text(
        json.dumps(
            {
                "10.shard/a": {"title": "from shard a"},
                "10.conflict/ref": {"title": "shard loses"},
            }
        ),
        encoding="utf-8",
    )
    (shard_a / "molecule_cache.json").write_text("{}", encoding="utf-8")
    (shard_b / "reference_cache.json").write_text(
        json.dumps({"10.shard/b": {"title": "from shard b"}}),
        encoding="utf-8",
    )
    (shard_b / "molecule_cache.json").write_text(
        json.dumps({"pubchem:shard": {"source": "shard b"}}),
        encoding="utf-8",
    )

    summary = merge_evidence_cache_shards(
        canonical_cache_dir=canonical,
        shard_cache_dirs=[shard_b, shard_a],
        conflict_policy="keep-canonical",
    )

    reference_cache = json.loads(
        (canonical / "reference_cache.json").read_text(encoding="utf-8")
    )
    molecule_cache = json.loads(
        (canonical / "molecule_cache.json").read_text(encoding="utf-8")
    )
    assert reference_cache["10.existing/canonical"]["title"] == "canonical existing"
    assert reference_cache["10.conflict/ref"]["title"] == "canonical wins"
    assert reference_cache["10.shard/a"]["title"] == "from shard a"
    assert reference_cache["10.shard/b"]["title"] == "from shard b"
    assert molecule_cache["pubchem:existing"]["source"] == "canonical"
    assert molecule_cache["pubchem:shard"]["source"] == "shard b"
    assert summary["merged_counts"] == {"reference": 2, "molecule": 1}
    assert summary["conflict_counts"] == {"reference": 1, "molecule": 0}
    assert summary["conflicts"] == [
        {
            "cache_file": "reference_cache.json",
            "key": "10.conflict/ref",
            "kept_source": str(canonical / "reference_cache.json"),
            "discarded_source": str(shard_a / "reference_cache.json"),
            "kept_sha256": summary["conflicts"][0]["kept_sha256"],
            "discarded_sha256": summary["conflicts"][0]["discarded_sha256"],
        }
    ]


def test_merge_evidence_cache_shards_fails_on_conflict_by_default(tmp_path):
    from data.evidence_cache_collector import merge_evidence_cache_shards

    canonical = tmp_path / "canonical"
    shard = tmp_path / "shard"
    canonical.mkdir()
    shard.mkdir()
    (canonical / "reference_cache.json").write_text(
        json.dumps({"10.conflict/ref": {"title": "canonical"}}),
        encoding="utf-8",
    )
    (shard / "reference_cache.json").write_text(
        json.dumps({"10.conflict/ref": {"title": "shard"}}),
        encoding="utf-8",
    )

    summary = merge_evidence_cache_shards(
        canonical_cache_dir=canonical,
        shard_cache_dirs=[shard],
        entity_type="reference",
    )

    reference_cache = json.loads(
        (canonical / "reference_cache.json").read_text(encoding="utf-8")
    )
    assert summary["merge_status"] == "failed"
    assert summary["conflict_counts"] == {"reference": 1, "molecule": 0}
    assert summary["conflicts"][0]["key"] == "10.conflict/ref"
    assert reference_cache["10.conflict/ref"]["title"] == "canonical"


def test_merge_evidence_cache_shards_requires_expected_shard_cache_files(tmp_path):
    from data.evidence_cache_collector import merge_evidence_cache_shards

    canonical = tmp_path / "canonical"
    shard = tmp_path / "missing-reference-shard"
    canonical.mkdir()
    shard.mkdir()
    (canonical / "reference_cache.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="reference_cache.json"):
        merge_evidence_cache_shards(
            canonical_cache_dir=canonical,
            shard_cache_dirs=[shard],
            entity_type="reference",
        )


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
        lambda: (
            lambda doi: ReferenceEvidence(
                doi=doi, title="CLI resolved", source="fixture-crossref"
            )
        ),
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


def test_collector_cli_accepts_shard_arguments():
    import run_evidence_cache_collector as runner

    args = runner.parse_args(
        [
            "--requirements-csv",
            "/tmp/requirements.csv",
            "--dataset-id",
            "collector-cli-shard",
            "--cache-dir",
            "/tmp/cache-shard-1",
            "--max-requests",
            "5",
            "--output-json",
            "/tmp/report.json",
            "--dry-run",
            "--shard-count",
            "3",
            "--shard-index",
            "1",
        ]
    )

    assert args.shard_count == 3
    assert args.shard_index == 1


def test_merge_cli_runs_against_temp_cache_dirs(tmp_path):
    import run_evidence_cache_merge as runner

    canonical = tmp_path / "canonical"
    shard = tmp_path / "shard"
    output_json = tmp_path / "merge-report.json"
    canonical.mkdir()
    shard.mkdir()
    (canonical / "reference_cache.json").write_text("{}", encoding="utf-8")
    (shard / "reference_cache.json").write_text(
        json.dumps({"10.shard/ref": {"title": "from shard"}}),
        encoding="utf-8",
    )

    exit_code = runner.main(
        [
            "--canonical-cache-dir",
            str(canonical),
            "--shard-cache-dir",
            str(shard),
            "--entity-type",
            "reference",
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    reference_cache = json.loads(
        (canonical / "reference_cache.json").read_text(encoding="utf-8")
    )
    assert report["merged_counts"]["reference"] == 1
    assert reference_cache["10.shard/ref"]["title"] == "from shard"


def test_merge_cli_writes_conflict_audit_and_returns_nonzero(tmp_path):
    import run_evidence_cache_merge as runner

    canonical = tmp_path / "canonical"
    shard = tmp_path / "shard"
    output_json = tmp_path / "merge-report.json"
    canonical.mkdir()
    shard.mkdir()
    (canonical / "reference_cache.json").write_text(
        json.dumps({"10.conflict/ref": {"title": "canonical"}}),
        encoding="utf-8",
    )
    (shard / "reference_cache.json").write_text(
        json.dumps({"10.conflict/ref": {"title": "shard"}}),
        encoding="utf-8",
    )

    exit_code = runner.main(
        [
            "--canonical-cache-dir",
            str(canonical),
            "--shard-cache-dir",
            str(shard),
            "--entity-type",
            "reference",
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    reference_cache = json.loads(
        (canonical / "reference_cache.json").read_text(encoding="utf-8")
    )
    assert exit_code == 1
    assert report["merge_status"] == "failed"
    assert report["conflicts"][0]["key"] == "10.conflict/ref"
    assert reference_cache["10.conflict/ref"]["title"] == "canonical"


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
        return ReferenceEvidence(
            doi=doi, title="Recovered reference", source="fixture-crossref"
        )

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
    reference_cache = json.loads(
        (cache_dir / "reference_cache.json").read_text(encoding="utf-8")
    )
    assert (
        reference_cache["10.1021/acs.jpclett.6c00119"]["title"] == "Recovered reference"
    )


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
    reference_cache = json.loads(
        (cache_dir / "reference_cache.json").read_text(encoding="utf-8")
    )
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
    reference_cache = json.loads(
        (cache_dir / "reference_cache.json").read_text(encoding="utf-8")
    )
    assert "10.1021/acs.jpclett.6c00119" not in reference_cache


def test_collector_never_writes_negative_cache_for_smiles_with_cid_only_resolver(
    tmp_path,
):
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
    molecule_cache = json.loads(
        (cache_dir / "molecule_cache.json").read_text(encoding="utf-8")
    )
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
        return ReferenceEvidence(
            doi=doi, title="Resolved reference", source="fixture-crossref"
        )

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
    assert snapshots[0]["entity_type_counts"] == {"reference": 1}
    assert snapshots[1]["entity_type_counts"] == {"reference": 1}
    assert snapshots[2]["entity_type_counts"] == {"reference": 1, "molecule": 1}
    assert snapshots[0]["positive_written_count"] == 1
    assert snapshots[1]["error_count"] == 1
    assert snapshots[2]["no_evidence_count"] == 1


def test_collector_cli_prints_progress_and_updates_report(
    tmp_path, monkeypatch, capsys
):
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
        lambda: (
            lambda doi: ReferenceEvidence(
                doi=doi, title="CLI resolved", source="fixture-crossref"
            )
        ),
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


def test_collector_progress_snapshot_cannot_mutate_final_summary(tmp_path):
    from data.evidence_cache_collector import collect_evidence_cache
    from harness.authenticity import ReferenceEvidence

    requirements = tmp_path / "requirements.csv"
    cache_dir = tmp_path / "evidence_cache"
    _requirements_csv(requirements)
    _seed_caches(cache_dir)

    def mutate_snapshot(snapshot):
        snapshot["processed"][0]["cache_status"] = "mutated"
        snapshot["processed"][0]["record_ids"].append("mutated-row")
        snapshot["entity_type_counts"]["reference"] = 999

    summary = collect_evidence_cache(
        requirements_csv=requirements,
        cache_dir=cache_dir,
        max_requests=1,
        progress_every=1,
        progress_callback=mutate_snapshot,
        reference_resolver=lambda doi: ReferenceEvidence(
            doi=doi,
            title="Resolved reference",
            source="fixture-crossref",
        ),
        molecule_resolver=lambda record: None,
    )

    assert summary["processed"][0]["cache_status"] == "positive"
    assert summary["processed"][0]["record_ids"] == ["row-001", "row-002"]
    assert summary["entity_type_counts"] == {"reference": 1}


def test_cache_write_failure_keeps_existing_json_readable(tmp_path, monkeypatch):
    from data.evidence_cache_collector import _write_cache

    cache_path = tmp_path / "reference_cache.json"
    cache_path.write_text(json.dumps({"existing": {"title": "safe"}}), encoding="utf-8")
    original_write_text = Path.write_text

    def failing_write_text(path, text, *args, **kwargs):
        original_write_text(path, '{"partial"', *args, **kwargs)
        raise RuntimeError("simulated interrupted write")

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    try:
        _write_cache({"new": {"title": "unsafe"}}, cache_path)
    except RuntimeError:
        pass

    assert json.loads(cache_path.read_text(encoding="utf-8")) == {
        "existing": {"title": "safe"}
    }


def test_report_write_failure_keeps_existing_json_readable(tmp_path, monkeypatch):
    from data.evidence_cache_collector import write_collection_report

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"attempted_count": 1}), encoding="utf-8")
    original_write_text = Path.write_text

    def failing_write_text(path, text, *args, **kwargs):
        original_write_text(path, '{"partial"', *args, **kwargs)
        raise RuntimeError("simulated interrupted write")

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    try:
        write_collection_report({"attempted_count": 2}, report_path)
    except RuntimeError:
        pass

    assert json.loads(report_path.read_text(encoding="utf-8")) == {"attempted_count": 1}


def test_atomic_write_cleans_temp_file_after_keyboard_interrupt(tmp_path, monkeypatch):
    from data.evidence_cache_collector import _write_cache

    cache_path = tmp_path / "reference_cache.json"
    cache_path.write_text(json.dumps({"existing": {"title": "safe"}}), encoding="utf-8")
    temp_path = cache_path.with_name(f".{cache_path.name}.tmp")
    original_write_text = Path.write_text

    def interrupted_write_text(path, text, *args, **kwargs):
        original_write_text(path, '{"partial"', *args, **kwargs)
        raise KeyboardInterrupt()

    monkeypatch.setattr(Path, "write_text", interrupted_write_text)

    try:
        _write_cache({"new": {"title": "unsafe"}}, cache_path)
    except KeyboardInterrupt:
        pass

    assert json.loads(cache_path.read_text(encoding="utf-8")) == {
        "existing": {"title": "safe"}
    }
    assert not temp_path.exists()
