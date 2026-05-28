#!/usr/bin/env python3
"""Audit repository-level single-source-of-truth conventions."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SOURCE_TABLE = (
    "/share/yhm/test/20260216_with_chemical_merge_fast/"
    "merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx"
)
CANONICAL_RUNNER = "hybrid_agent_exploration/src/run_research_package.py"
CANONICAL_OUTPUT_ROOT = "hybrid_agent_exploration/results/verified_discovery_runs"
SOURCE_OF_TRUTH_DOC = "SOURCE_OF_TRUTH.md"

STALE_README_TOKENS = (
    "data/raw/perovskite_database_all.csv",
    "python src/eda/analyze.py",
    "python src/automl/train.py",
    "R² > 0.85",
)

REGENERABLE_TRACKED_PREFIXES = (
    "autogluon_models/",
    "figures/",
    "fingerprints/",
    "ML-Master/",
    "tables/",
    "reports/",
    "report/",
)

REGENERABLE_TRACKED_FILES = (
    "data/processed/perovskite_cleaned.csv",
    "data/processed/perovskite_ml_ready.csv",
    "processed_data.csv",
)


def _read_text(root: Path, relative_path: str) -> str:
    return (root / relative_path).read_text(encoding="utf-8")


def _git_tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return []
    return result.stdout.splitlines()


def audit_repository(root: Path = ROOT) -> list[str]:
    """Return source-of-truth issues for the repository root."""
    root = root.resolve()
    issues: list[str] = []

    readme = _read_text(root, "README.md")
    makefile = _read_text(root, "Makefile")

    source_doc_path = root / SOURCE_OF_TRUTH_DOC
    if not source_doc_path.exists():
        issues.append(f"missing canonical source-of-truth document: {SOURCE_OF_TRUTH_DOC}")
        source_doc = ""
    else:
        source_doc = source_doc_path.read_text(encoding="utf-8")

    expected_makefile_lines = (
        f"CANONICAL_SOURCE_TABLE ?= {CANONICAL_SOURCE_TABLE}",
        "SOURCE_TABLE ?= $(CANONICAL_SOURCE_TABLE)",
        f"RUN_RESEARCH_PACKAGE := {CANONICAL_RUNNER}",
        f"ARTIFACT_DIR ?= {CANONICAL_OUTPUT_ROOT}/$(DATASET_ID)",
    )
    for line in expected_makefile_lines:
        if line not in makefile:
            issues.append(f"Makefile does not declare canonical contract line: {line}")

    required_readme_tokens = (
        CANONICAL_SOURCE_TABLE,
        CANONICAL_RUNNER,
        "make research-package",
        "make research-package-smoke",
        "make test-research-package",
        SOURCE_OF_TRUTH_DOC,
    )
    for token in required_readme_tokens:
        if token not in readme:
            issues.append(f"README.md is missing canonical token: {token}")

    for token in STALE_README_TOKENS:
        if token in readme:
            issues.append(f"README.md still advertises stale workflow token: {token}")

    required_source_doc_tokens = (
        CANONICAL_SOURCE_TABLE,
        CANONICAL_RUNNER,
        CANONICAL_OUTPUT_ROOT,
        "historical artifacts",
        "Makefile",
    )
    for token in required_source_doc_tokens:
        if token not in source_doc:
            issues.append(f"{SOURCE_OF_TRUTH_DOC} is missing canonical token: {token}")

    tracked_files = _git_tracked_files(root)
    for path in tracked_files:
        if path in REGENERABLE_TRACKED_FILES:
            issues.append(f"regenerable data artifact is tracked: {path}")
        if path.startswith("data/processed/") and path.endswith(".csv"):
            issues.append(f"processed CSV export is tracked: {path}")
        if path.startswith(REGENERABLE_TRACKED_PREFIXES):
            issues.append(f"legacy generated artifact is tracked: {path}")

    return sorted(set(issues))


def main() -> int:
    issues = audit_repository()
    if issues:
        print("Source-of-truth audit failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Source-of-truth audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
