#!/bin/bash
# ============================================================================
# PSC ML All-in-One — 先跑探索再生成报告
#
# 用法:
#   bash run_all.sh          # 默认跑 20 轮 + 生成报告
#   bash run_all.sh 50       # 跑 50 轮 + 生成报告
# ============================================================================

set -e

N_RUNS="${1:-20}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  PSC ML Pipeline Exploration"
echo "  Phase 1: ${N_RUNS} random explorations"
echo "  Phase 2: Scientific report generation"
echo "============================================"

# Phase 1: Exploration
echo ""
echo ">>> Phase 1: Running ${N_RUNS} explorations..."
bash "${SCRIPT_DIR}/run_explore.sh" "${N_RUNS}"

# Phase 2: Report
echo ""
echo ">>> Phase 2: Generating scientific report..."
bash "${SCRIPT_DIR}/run_report.sh"

echo ""
echo "============================================"
echo "  Done!"
echo "  Results: ${SCRIPT_DIR}/results/exploration_logs/"
echo "  Figures: ${SCRIPT_DIR}/results/figures/"
echo "  Report:  ${SCRIPT_DIR}/results/comparison_reports/scientific_report.md"
echo "============================================"
