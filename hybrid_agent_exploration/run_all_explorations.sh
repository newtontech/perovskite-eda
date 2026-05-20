#!/usr/bin/env bash
# =============================================================================
# run_all_explorations.sh
# Master runner for all five AGENTS.md layer explorations.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPLORATIONS_DIR="${SCRIPT_DIR}/explorations"
LOG_DIR="${SCRIPT_DIR}/results/exploration_logs"
mkdir -p "${LOG_DIR}"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/exploration_run_${TIMESTAMP}.log"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${MASTER_LOG}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${MASTER_LOG}"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "${MASTER_LOG}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${MASTER_LOG}"
}

run_exploration() {
    local name="$1"
    local script_path="$2"
    local log_file="${LOG_DIR}/${name}_${TIMESTAMP}.log"
    
    log_info "========================================"
    log_info "Running: ${name}"
    log_info "Script: ${script_path}"
    log_info "Log: ${log_file}"
    log_info "========================================"
    
    if [[ ! -f "${script_path}" ]]; then
        log_error "Script not found: ${script_path}"
        return 1
    fi
    
    local script_dir
    script_dir=$(dirname "${script_path}")
    
    (
        cd "${script_dir}" || exit 1
        if python "${script_path}" > "${log_file}" 2>&1; then
            log_success "${name} completed successfully."
            return 0
        else
            log_error "${name} failed. Check log: ${log_file}"
            return 1
        fi
    )
}

# =============================================================================
# Main
# =============================================================================

log_info "Starting Hybrid Agent Exploration Run: ${TIMESTAMP}"
log_info "Project: ${SCRIPT_DIR}"
log_info ""

# ---------------------------------------------------------------------------
# Exploration 1: Layer 1 — Data Source Exploration
# ---------------------------------------------------------------------------
run_exploration \
    "data_source_exploration" \
    "${EXPLORATIONS_DIR}/data_source_exploration/explore_data_sources.py"

# ---------------------------------------------------------------------------
# Exploration 2: Layer 2 — Feature Engineering Exploration
# ---------------------------------------------------------------------------
run_exploration \
    "feature_engineering_exploration" \
    "${EXPLORATIONS_DIR}/feature_engineering_exploration/explore_features.py"

# ---------------------------------------------------------------------------
# Layer 3 — Model Architecture Exploration
# ---------------------------------------------------------------------------
run_exploration \
    "model_architecture_exploration" \
    "${EXPLORATIONS_DIR}/model_architecture_exploration/explore_models.py"

# ---------------------------------------------------------------------------
# Layer 4 — Evaluation Strategy Exploration
# ---------------------------------------------------------------------------
run_exploration \
    "evaluation_strategy_exploration" \
    "${EXPLORATIONS_DIR}/evaluation_strategy_exploration/explore_evaluation.py"

# ---------------------------------------------------------------------------
# Layer 5 — Deployment & Screening Exploration
# ---------------------------------------------------------------------------
run_exploration \
    "deployment_screening_exploration" \
    "${EXPLORATIONS_DIR}/deployment_screening_exploration/explore_screening.py"

# =============================================================================
# Summary
# =============================================================================
log_info ""
log_info "========================================"
log_info "All explorations completed."
log_info "Master log: ${MASTER_LOG}"
log_info "Individual logs: ${LOG_DIR}/"
log_info "========================================"

# Optionally generate a summary report
echo ""
echo "Exploration outputs:"
find "${EXPLORATIONS_DIR}" -maxdepth 2 -type f \( -name "*.csv" -o -name "*.json" -o -name "*.png" \) | sort | while read -r f; do
    echo "  - ${f}"
done
