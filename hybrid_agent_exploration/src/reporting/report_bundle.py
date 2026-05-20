"""Typed return value for generated report artifacts."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ReportBundle:
    """Paths and quality metadata emitted by a report generator."""

    path: Path
    figures: list[Path] = field(default_factory=list)
    claim_ledger: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    manifest_path: Path | None = None
    review_path: Path | None = None
