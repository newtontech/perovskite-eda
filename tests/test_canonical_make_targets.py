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
        "research-package-pdf",
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


def test_test_research_package_target_runs_canonical_pytest_slice():
    output = _make_dry_run("test-research-package")

    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in output
    assert "tests/test_canonical_make_targets.py" in output
    assert "tests/test_research_package_runner.py" in output
    assert "tests/test_run_verified_discovery_cli.py" in output
    assert "tests/test_root_provenance_manifest.py" in output
