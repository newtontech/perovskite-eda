import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HYBRID_ROOT = PROJECT_ROOT / "hybrid_agent_exploration"
sys.path.insert(0, str(HYBRID_ROOT / "scripts"))

from audit_cleanup import audit_tree


def test_hybrid_active_tree_stays_trimmed_to_source_and_key_outputs():
    assert audit_tree(HYBRID_ROOT) == []
