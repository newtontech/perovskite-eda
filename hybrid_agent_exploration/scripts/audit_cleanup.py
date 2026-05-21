#!/usr/bin/env python3
"""Audit the active hybrid-agent workspace for regenerable bulk artifacts."""

from __future__ import annotations

from pathlib import Path


HYBRID_ROOT = Path(__file__).resolve().parents[1]
MAX_ACTIVE_FILE_BYTES = 5 * 1024 * 1024

ALLOWED_LARGE_FILES = {
    Path("article/manuscript_v2.pdf"),
}

ALLOWED_LARGE_PREFIXES = (
    "ref/",
)

FORBIDDEN_PATHS = (
    Path("results/report_artifacts"),
    Path("results/harness_logs"),
    Path("explorations/data_source_exploration/.cache"),
    Path("article/figures"),
)

FORBIDDEN_GLOBS = (
    "results/research_stories/minimal_*",
    "results/research_stories/jpcl_sam_run_*",
    "results/research_stories/top_journal_pr_validation",
    "article/preview-*.png",
    "article/preview_fig-*.png",
    "article/v2_preview*.png",
    "article/manuscript_with_figures.html",
    "article/manuscript_with_figures.md",
    "article/manuscript.md",
    "article/manuscript.pdf",
    "explorations/data_source_exploration/*cleaned*.csv",
    "explorations/data_source_exploration/agentic_evolution_best.csv",
)


def _rel(root: Path, path: Path) -> Path:
    return path.relative_to(root)


def audit_tree(root: Path = HYBRID_ROOT) -> list[str]:
    """Return cleanup issues for the active tree."""
    root = root.resolve()
    issues: list[str] = []

    for rel_path in FORBIDDEN_PATHS:
        path = root / rel_path
        if path.exists():
            issues.append(f"bulk artifact remains: {rel_path.as_posix()}")

    for pattern in FORBIDDEN_GLOBS:
        for path in root.glob(pattern):
            issues.append(f"bulk artifact remains: {_rel(root, path).as_posix()}")

    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "data:image" in text or ";base64," in text:
            issues.append(f"base64 image embedded in markdown: {_rel(root, path).as_posix()}")

    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size <= MAX_ACTIVE_FILE_BYTES:
            continue
        rel_path = _rel(root, path)
        rel_posix = rel_path.as_posix()
        if rel_path in ALLOWED_LARGE_FILES:
            continue
        if rel_posix.startswith(ALLOWED_LARGE_PREFIXES):
            continue
        issues.append(f"unapproved large file: {rel_posix} ({path.stat().st_size} bytes)")

    return sorted(issues)


def main() -> int:
    issues = audit_tree()
    if issues:
        print("Hybrid cleanup audit failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Hybrid cleanup audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
