#!/usr/bin/env bash
# =============================================================================
# multi_agent_run.sh
# Convenience runner for multi-agent parallel cross-layer exploration.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Defaults
N_AGENTS="${N_AGENTS:-12}"
MAX_WORKERS="${MAX_WORKERS:-4}"
TARGET="${TARGET:-delta_pce}"
OUTPUT="${OUTPUT:-results/multi_agent_exploration}"

echo "========================================"
echo "Multi-Agent Cross-Layer Exploration"
echo "========================================"
echo "Agents:      ${N_AGENTS}"
echo "Workers:     ${MAX_WORKERS}"
echo "Target:      ${TARGET}"
echo "Output:      ${OUTPUT}"
echo "========================================"

cd "${PROJECT_ROOT}"

# Run orchestrator
python src/orchestrator.py \
  --n-agents "${N_AGENTS}" \
  --max-workers "${MAX_WORKERS}" \
  --target "${TARGET}" \
  --baseline-as-feature \
  --weighted-sampling \
  --fast-only \
  --output "${OUTPUT}" \
  --checkpoint-every 5

# Run analysis
echo ""
echo "Running post-hoc analysis ..."
python src/exploration_analyzer.py --input "${OUTPUT}"

echo ""
echo "========================================"
echo "Exploration complete."
echo "Results: ${PROJECT_ROOT}/${OUTPUT}"
echo "========================================"
