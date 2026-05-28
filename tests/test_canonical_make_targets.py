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
        "audit-source-of-truth",
        "research-package",
        "research-package-smoke",
        "research-package-cache-preflight",
        "research-package-cache-preflight-smoke",
        "research-package-cache-collect",
        "research-package-cache-collect-dry-run",
        "research-package-cache-merge",
        "research-package-psc-candidate-closure",
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
    assert (
        "hybrid_agent_exploration/results/verified_discovery_runs/unit-dataset"
        in output
    )


def test_psc_candidate_closure_target_uses_real_seed_source_and_strict_verify():
    output = _make_dry_run("research-package-psc-candidate-closure")
    cache_dir = (
        ROOT
        / "hybrid_agent_exploration"
        / ".cache"
        / "verified_discovery"
        / "psc-additive-closure-external-cached-full-source"
        / "evidence_cache"
    )

    assert "research-package" in output
    assert "research-package-pdf" in output
    assert "research-package-verify" in output
    assert "DATASET_ID=psc-additive-closure-candidate-loop" in output
    assert "EVIDENCE_MODE=external-cached" in output
    assert "INPUT_SCOPE=full-source" in output
    assert f"test -f {cache_dir}/reference_cache.json" in output
    assert f"test -f {cache_dir}/molecule_cache.json" in output
    assert f"--cache-dir {cache_dir}" in output
    assert (
        "CANDIDATE_SOURCE=hybrid_agent_exploration/candidate_sources/"
        "psc_passivator_seed_candidates.csv"
        in output
    )
    assert "CANDIDATE_SOURCE_NAME=psc-passivator-seed" in output
    assert "REQUIRE_CANDIDATE_LIBRARY=1" in output
    assert "REQUIRE_EVIDENCE_CACHE=1" in output


def test_research_package_verify_dry_run_accepts_closure_readiness_gate():
    output = _make_dry_run(
        "research-package-verify",
        ARTIFACT_DIR="/tmp/artifacts",
        REQUIRE_CLOSURE_READINESS="1",
    )

    assert "hybrid_agent_exploration/src/verify_research_package.py" in output
    assert "--package-dir /tmp/artifacts" in output
    assert "--require-closure-readiness" in output


def test_research_package_dry_run_accepts_experimental_feedback_args():
    output = _make_dry_run(
        "research-package",
        SOURCE_TABLE="/tmp/source.csv",
        ARTIFACT_DIR="/tmp/artifacts",
        DATASET_ID="feedback-dataset",
        EXPERIMENTAL_FEEDBACK_CSV="/tmp/feedback.csv",
        EXPERIMENTAL_FEEDBACK_DATA_SCOPE="external_lab",
    )

    assert "hybrid_agent_exploration/src/run_research_package.py" in output
    assert "--experimental-feedback-csv /tmp/feedback.csv" in output
    assert "--experimental-feedback-data-scope external_lab" in output


def test_research_package_verify_dry_run_accepts_experimental_feedback_gate():
    output = _make_dry_run(
        "research-package-verify",
        ARTIFACT_DIR="/tmp/artifacts",
        REQUIRE_EXPERIMENTAL_FEEDBACK="1",
    )

    assert "hybrid_agent_exploration/src/verify_research_package.py" in output
    assert "--package-dir /tmp/artifacts" in output
    assert "--require-experimental-feedback" in output


def test_smoke_target_dry_run_caps_rows_in_ignored_artifact_dir():
    output = _make_dry_run("research-package-smoke", SOURCE_TABLE="/tmp/source.csv")

    assert "hybrid_agent_exploration/src/run_research_package.py" in output
    assert "--max-rows" in output
    assert "hybrid_agent_exploration/results/verified_discovery_runs/" in output


def test_source_of_truth_audit_target_runs_repo_guardrail():
    output = _make_dry_run("audit-source-of-truth")

    assert "scripts/audit_source_of_truth.py" in output


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
    assert (
        "--requirements-csv /tmp/artifacts/evidence_cache_preflight/evidence_cache_requirements.csv"
        in output
    )
    assert "--dataset-id unit-dataset" in output
    assert "--max-requests 7" in output
    assert "--progress-every 10" in output
    assert (
        "--output-json /tmp/artifacts/evidence_cache_collection_report.json" in output
    )
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


def test_cache_collect_target_accepts_shard_variables():
    output = _make_dry_run(
        "research-package-cache-collect",
        DATASET_ID="unit-dataset",
        ARTIFACT_DIR="/tmp/artifacts",
        CACHE_COLLECT_SHARD_COUNT="4",
        CACHE_COLLECT_SHARD_INDEX="2",
    )

    assert "--shard-count 4" in output
    assert "--shard-index 2" in output
    assert (
        "--cache-dir /tmp/artifacts/evidence_cache_shards/shard-2/evidence_cache"
        in output
    )
    assert (
        "--output-json /tmp/artifacts/evidence_cache_collection_shard_2_report.json"
        in output
    )


def test_cache_collect_shard_defaults_index_zero_to_shard_local_cache():
    output = _make_dry_run(
        "research-package-cache-collect",
        DATASET_ID="unit-dataset",
        ARTIFACT_DIR="/tmp/artifacts",
        CACHE_COLLECT_SHARD_COUNT="4",
    )

    assert "--shard-count 4" in output
    assert "--shard-index 0" in output
    assert (
        "--cache-dir /tmp/artifacts/evidence_cache_shards/shard-0/evidence_cache"
        in output
    )
    assert (
        "--output-json /tmp/artifacts/evidence_cache_collection_shard_0_report.json"
        in output
    )


def test_cache_collect_shard_ignores_canonical_evidence_cache_dir():
    output = _make_dry_run(
        "research-package-cache-collect",
        DATASET_ID="unit-dataset",
        ARTIFACT_DIR="/tmp/artifacts",
        EVIDENCE_CACHE_DIR="/tmp/canonical-cache",
        CACHE_COLLECT_SHARD_COUNT="4",
        CACHE_COLLECT_SHARD_INDEX="1",
    )

    assert (
        "--cache-dir /tmp/artifacts/evidence_cache_shards/shard-1/evidence_cache"
        in output
    )
    assert "--cache-dir /tmp/canonical-cache" not in output


def test_cache_merge_target_dry_run_merges_declared_shard_dirs():
    output = _make_dry_run(
        "research-package-cache-merge",
        DATASET_ID="unit-dataset",
        ARTIFACT_DIR="/tmp/artifacts",
        EVIDENCE_CACHE_DIR="/tmp/canonical-cache",
        CACHE_SHARD_DIRS="/tmp/shard-b /tmp/shard-a",
    )

    assert "hybrid_agent_exploration/src/run_evidence_cache_merge.py" in output
    assert "--canonical-cache-dir /tmp/canonical-cache" in output
    assert "--shard-cache-dir /tmp/shard-b" in output
    assert "--shard-cache-dir /tmp/shard-a" in output
    assert "--output-json /tmp/artifacts/evidence_cache_merge_report.json" in output


def test_cache_merge_target_requires_declared_shard_dirs():
    output = _make_dry_run(
        "research-package-cache-merge",
        DATASET_ID="unit-dataset",
        ARTIFACT_DIR="/tmp/artifacts",
    )

    assert "CACHE_SHARD_DIRS is required for research-package-cache-merge" in output
    assert "run_evidence_cache_merge.py" not in output


def test_pdf_target_dry_run_checks_pandoc_and_exports_report_pdfs():
    output = _make_dry_run("research-package-pdf", ARTIFACT_DIR="/tmp/artifacts")

    assert "command -v pandoc" in output
    assert "--pdf-engine=xelatex" in output
    assert '-V mainfont="DejaVu Serif"' in output
    assert (
        "--resource-path=/tmp/artifacts/report_bundle/main_text:/tmp/artifacts"
        in output
    )
    assert "main_text/main_text_report.md" in output
    assert "main_text/main_text_report.pdf" in output
    assert "si/supporting_information.md" in output
    assert "si/supporting_information.pdf" in output
    assert "pandoc" in output
    assert (
        "hybrid_agent_exploration/src/reporting/root_provenance_manifest.py" in output
    )
    assert "--output-path /tmp/artifacts/provenance_manifest.json" in output
    assert "--evidence-cache-manifest-path /tmp/artifacts/evidence_cache/evidence_cache_manifest.json" in output


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
    assert "tests/test_experimental_feedback_ingestion.py" in output
    assert "tests/test_research_package_runner.py" in output
    assert "tests/test_run_verified_discovery_cli.py" in output
    assert "tests/test_root_provenance_manifest.py" in output
