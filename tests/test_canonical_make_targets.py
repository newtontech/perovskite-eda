from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"


def _make_dry_run(target: str, **variables: str) -> str:
    args = ["make", "-n", target]
    args.extend(f"{key}={value}" for key, value in variables.items())
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    return result.stdout


def test_canonical_research_package_targets_are_declared():
    text = MAKEFILE.read_text(encoding="utf-8")

    for target in (
        "research-package",
        "research-package-smoke",
        "research-package-cache-preflight",
        "research-package-cache-preflight-smoke",
        "research-package-cache-collect",
        "research-package-cache-collect-dry-run",
        "research-package-pdf",
        "research-package-verify",
        "test-research-package",
    ):
        assert f".PHONY: {target}" in text or f" {target}" in text
        assert f"{target}:" in text


def test_research_package_dry_run_uses_canonical_runner_and_ignored_artifact_dir():
    output = _make_dry_run(
        "research-package",
        SOURCE_TABLE="/tmp/source.csv",
        DATASET_ID="unit-dataset",
        EVIDENCE_MODE="source-columns",
        INPUT_SCOPE="full-source",
        MIN_VERIFIED_ROWS="3",
        TOP_K="7",
        CANDIDATE_SOURCE="/tmp/candidates.csv",
        CANDIDATE_SOURCE_NAME="unit-candidates",
    )

    assert "hybrid_agent_exploration/src/run_research_package.py" in output
    assert "--input /tmp/source.csv" in output
    assert "--dataset-id unit-dataset" in output
    assert "--evidence-mode source-columns" in output
    assert "--input-scope full-source" in output
    assert "--min-verified-rows 3" in output
    assert "--top-k 7" in output
    assert "--candidate-source /tmp/candidates.csv" in output
    assert "--candidate-source-name unit-candidates" in output
    assert "hybrid_agent_exploration/results/verified_discovery_runs/unit-dataset" in output


def test_smoke_target_dry_run_caps_rows_in_ignored_artifact_dir():
    output = _make_dry_run("research-package-smoke", SOURCE_TABLE="/tmp/source.csv")

    assert "hybrid_agent_exploration/src/run_research_package.py" in output
    assert "--max-rows" in output
    assert "hybrid_agent_exploration/results/verified_discovery_runs/" in output


def test_cache_preflight_target_dry_run_scans_full_table_by_default():
    output = _make_dry_run(
        "research-package-cache-preflight",
        SOURCE_TABLE="/tmp/source.csv",
        DATASET_ID="unit-dataset",
        ARTIFACT_DIR="/tmp/artifacts",
    )

    assert "hybrid_agent_exploration/src/run_evidence_cache_preflight.py" in output
    assert "--input /tmp/source.csv" in output
    assert "--dataset-id unit-dataset" in output
    assert "--output-dir /tmp/artifacts/evidence_cache_preflight" in output
    assert "--cache-dir" not in output
    assert "--max-rows" not in output


def test_cache_preflight_target_accepts_explicit_cache_dir_and_row_cap():
    output = _make_dry_run(
        "research-package-cache-preflight",
        SOURCE_TABLE="/tmp/source.csv",
        DATASET_ID="unit-dataset",
        ARTIFACT_DIR="/tmp/artifacts",
        EVIDENCE_CACHE_DIR="/tmp/cache",
        CACHE_PREFLIGHT_MAX_ROWS="25",
    )

    assert "hybrid_agent_exploration/src/run_evidence_cache_preflight.py" in output
    assert "--input /tmp/source.csv" in output
    assert "--dataset-id unit-dataset" in output
    assert "--cache-dir /tmp/cache" in output
    assert "--output-dir /tmp/artifacts/evidence_cache_preflight" in output
    assert "--max-rows 25" in output


def test_cache_preflight_smoke_target_dry_run_caps_rows_explicitly():
    output = _make_dry_run(
        "research-package-cache-preflight-smoke",
        SOURCE_TABLE="/tmp/source.csv",
        CACHE_PREFLIGHT_SMOKE_MAX_ROWS="25",
    )

    assert "research-package-cache-preflight" in output
    assert "CACHE_PREFLIGHT_MAX_ROWS=25" in output


def test_cache_collect_target_dry_run_uses_preflight_requirements_and_budget():
    output = _make_dry_run(
        "research-package-cache-collect",
        DATASET_ID="unit-dataset",
        ARTIFACT_DIR="/tmp/artifacts",
        CACHE_COLLECT_MAX_REQUESTS="7",
    )

    assert "hybrid_agent_exploration/src/run_evidence_cache_collector.py" in output
    assert "--requirements-csv /tmp/artifacts/evidence_cache_preflight/evidence_cache_requirements.csv" in output
    assert "--dataset-id unit-dataset" in output
    assert "--max-requests 7" in output
    assert "--output-json /tmp/artifacts/evidence_cache_collection_report.json" in output
    assert "--cache-dir" not in output
    assert "--dry-run" not in output


def test_cache_collect_dry_run_target_does_not_write_cache():
    output = _make_dry_run(
        "research-package-cache-collect-dry-run",
        DATASET_ID="unit-dataset",
        CACHE_COLLECT_MAX_REQUESTS="7",
    )

    assert "research-package-cache-collect" in output
    assert "EXTRA_CACHE_COLLECT_ARGS=--dry-run" in output


def test_pdf_target_dry_run_checks_pandoc_and_exports_report_pdfs():
    output = _make_dry_run("research-package-pdf", ARTIFACT_DIR="/tmp/artifacts")

    assert "command -v pandoc" in output
    assert "--pdf-engine=xelatex" in output
    assert '-V mainfont="DejaVu Serif"' in output
    assert "--resource-path=/tmp/artifacts/report_bundle/main_text:/tmp/artifacts" in output
    assert "main_text/main_text_report.md" in output
    assert "main_text/main_text_report.pdf" in output
    assert "si/supporting_information.md" in output
    assert "si/supporting_information.pdf" in output
    assert "pandoc" in output
    assert "hybrid_agent_exploration/src/reporting/root_provenance_manifest.py" in output
    assert "--output-path /tmp/artifacts/provenance_manifest.json" in output


def test_verify_target_dry_run_runs_contract_verifier_with_optional_gates():
    output = _make_dry_run(
        "research-package-verify",
        ARTIFACT_DIR="/tmp/artifacts",
        REQUIRE_CANDIDATE_LIBRARY="1",
        REQUIRE_EVIDENCE_CACHE="1",
    )

    assert "hybrid_agent_exploration/src/verify_research_package.py" in output
    assert "--package-dir /tmp/artifacts" in output
    assert "--require-candidate-library" in output
    assert "--require-evidence-cache" in output


def test_test_research_package_target_runs_canonical_pytest_slice():
    output = _make_dry_run("test-research-package")

    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in output
    assert "tests/test_canonical_make_targets.py" in output
    assert "tests/test_evidence_cache_collector.py" in output
    assert "tests/test_evidence_cache_preflight.py" in output
    assert "tests/test_research_package_runner.py" in output
    assert "tests/test_run_verified_discovery_cli.py" in output
    assert "tests/test_root_provenance_manifest.py" in output
