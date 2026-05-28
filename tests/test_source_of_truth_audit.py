from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_source_of_truth import audit_repository


def test_repository_declares_one_canonical_research_source_and_entrypoint():
    assert audit_repository(ROOT) == []
