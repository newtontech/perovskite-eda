import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _source_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "row-001",
                "doi": "10.1021/ACS.JPCLETT.6C00119",
                "title": "Verified PSC additive paper",
                "smiles": "[Ba+2]",
                "pubchem_id": "104810.0",
                "cas_number": "22541-12-4",
            },
            {
                "record_id": "row-002",
                "doi": "10.1021/acs.jpclett.6c00119",
                "title": "Duplicate DOI and CID",
                "smiles": "[Ba+2]",
                "pubchem_id": "104810",
                "cas_number": "22541-12-4",
            },
            {
                "record_id": "row-003",
                "doi": "10.1038/s41586-026-00001-1",
                "title": "Negative cached reference",
                "smiles": "CCO",
                "pubchem_id": "",
                "cas_number": "64-17-5",
            },
            {
                "record_id": "row-004",
                "doi": "10.1126/science.abd0001",
                "title": "Missing reference cache entry",
                "smiles": "CCO",
                "pubchem_id": "",
                "cas_number": "64-17-5",
            },
            {
                "record_id": "row-005",
                "doi": "",
                "title": "Uncacheable row",
                "smiles": "",
                "pubchem_id": "",
                "cas_number": "",
            },
        ]
    )


def _write_cache_files(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True)
    (cache_dir / "reference_cache.json").write_text(
        json.dumps(
            {
                "10.1021/acs.jpclett.6c00119": {
                    "kind": "reference",
                    "source": "crossref",
                    "doi": "10.1021/acs.jpclett.6c00119",
                    "title": "Verified PSC additive paper",
                    "year": 2026,
                    "journal": "Journal of Physical Chemistry Letters",
                    "url": "https://doi.org/10.1021/acs.jpclett.6c00119",
                },
                "10.1038/s41586-026-00001-1": None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (cache_dir / "molecule_cache.json").write_text(
        json.dumps(
            {
                "pubchem:104810": {
                    "kind": "molecule",
                    "source": "pubchem",
                    "smiles": "[Ba+2]",
                    "pubchem_id": "104810",
                    "cas_number": "22541-12-4",
                    "url": "https://pubchem.ncbi.nlm.nih.gov/compound/104810",
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_summarizes_required_external_cache_coverage(tmp_path):
    from data.evidence_cache_preflight import summarize_evidence_cache_preflight

    cache_dir = tmp_path / "evidence_cache"
    _write_cache_files(cache_dir)

    summary = summarize_evidence_cache_preflight(
        _source_dataframe(),
        dataset_id="cache-preflight-fixture",
        source_name="fixture-source",
        cache_dir=cache_dir,
    )

    assert summary["schema_version"] == "evidence-cache-preflight-v1"
    assert summary["dataset_id"] == "cache-preflight-fixture"
    assert summary["source_name"] == "fixture-source"
    assert summary["cache_dir"] == str(cache_dir)
    assert summary["row_count"] == 5
    assert summary["external_cache_ready"] is False
    assert summary["all_positive_evidence_cached"] is False

    assert summary["reference_cache"]["required_count"] == 3
    assert summary["reference_cache"]["cached_count"] == 2
    assert summary["reference_cache"]["positive_cached_count"] == 1
    assert summary["reference_cache"]["negative_cached_count"] == 1
    assert summary["reference_cache"]["missing_count"] == 1
    assert summary["reference_cache"]["uncacheable_row_count"] == 1
    assert summary["reference_cache"]["missing_keys"] == ["10.1126/science.abd0001"]

    assert summary["molecule_cache"]["required_count"] == 2
    assert summary["molecule_cache"]["cached_count"] == 1
    assert summary["molecule_cache"]["positive_cached_count"] == 1
    assert summary["molecule_cache"]["negative_cached_count"] == 0
    assert summary["molecule_cache"]["missing_count"] == 1
    assert summary["molecule_cache"]["uncacheable_row_count"] == 1
    assert summary["molecule_cache"]["missing_keys"] == ["smiles:CCO"]

    assert summary["requirements"][0]["entity_type"] == "reference"
    assert summary["requirements"][0]["key"] == "10.1021/acs.jpclett.6c00119"
    assert summary["requirements"][0]["row_count"] == 2
    assert summary["requirements"][0]["cache_status"] == "positive"
    assert summary["requirements"][0]["record_ids"] == ["row-001", "row-002"]


def test_preflight_writes_json_csv_and_markdown_artifacts(tmp_path):
    from data.evidence_cache_preflight import (
        summarize_evidence_cache_preflight,
        write_evidence_cache_preflight_artifacts,
    )

    cache_dir = tmp_path / "evidence_cache"
    _write_cache_files(cache_dir)
    output_dir = tmp_path / "preflight"

    artifacts = write_evidence_cache_preflight_artifacts(
        summarize_evidence_cache_preflight(
            _source_dataframe(),
            dataset_id="cache-preflight-artifacts",
            source_name="fixture-source",
            cache_dir=cache_dir,
        ),
        output_dir,
    )

    assert artifacts.output_dir == output_dir
    assert artifacts.summary_json == output_dir / "evidence_cache_preflight.json"
    assert artifacts.requirements_csv == output_dir / "evidence_cache_requirements.csv"
    assert artifacts.report_md == output_dir / "evidence_cache_preflight.md"

    summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    assert summary["dataset_id"] == "cache-preflight-artifacts"
    assert summary["outputs"]["summary_json"] == "evidence_cache_preflight.json"
    assert summary["outputs"]["requirements_csv"] == "evidence_cache_requirements.csv"
    assert summary["outputs"]["report_md"] == "evidence_cache_preflight.md"

    requirements = pd.read_csv(artifacts.requirements_csv)
    assert set(requirements["entity_type"]) == {"reference", "molecule"}
    assert "cache_status" in requirements.columns
    assert requirements.loc[requirements["key"] == "smiles:CCO", "cache_status"].item() == "missing"

    report = artifacts.report_md.read_text(encoding="utf-8")
    assert "# External Evidence Cache Preflight" in report
    assert "Reference cache coverage" in report
    assert "Molecule cache coverage" in report
    assert "`smiles:CCO`" in report


def test_preflight_cli_writes_artifacts_without_network_access(tmp_path, monkeypatch):
    from run_evidence_cache_preflight import main
    import harness.authenticity as authenticity

    source_csv = tmp_path / "source.csv"
    cache_dir = tmp_path / "evidence_cache"
    output_dir = tmp_path / "preflight"
    _source_dataframe().to_csv(source_csv, index=False)
    _write_cache_files(cache_dir)
    cache_fingerprints = {
        path.name: (path.stat().st_mtime_ns, _sha256(path))
        for path in (cache_dir / "reference_cache.json", cache_dir / "molecule_cache.json")
    }

    def fail_on_network(*args, **kwargs):
        raise AssertionError("preflight must not call external network resolvers")

    monkeypatch.setattr(authenticity.requests, "get", fail_on_network)

    exit_code = main(
        [
            "--input",
            str(source_csv),
            "--dataset-id",
            "cache-preflight-cli",
            "--source-name",
            "fixture-source",
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "evidence_cache_preflight.json").read_text(encoding="utf-8"))
    assert summary["dataset_id"] == "cache-preflight-cli"
    assert summary["network_access"] == "not_used"
    assert summary["reference_cache"]["missing_count"] == 1
    assert summary["molecule_cache"]["missing_count"] == 1
    assert {
        path.name: (path.stat().st_mtime_ns, _sha256(path))
        for path in (cache_dir / "reference_cache.json", cache_dir / "molecule_cache.json")
    } == cache_fingerprints
