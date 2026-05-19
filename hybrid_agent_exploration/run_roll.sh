#!/bin/bash
# ============================================================================
# PSC ML Single Roll — 摇一次骰子，随机选一条管线执行
#
# 用法:
#   bash run_roll.sh
# ============================================================================

set -e

PROJECT_DIR="/share/yhm/test/AutoML_EDA/hybrid_agent_exploration"
MODEL="openrouter/moonshotai/kimi-k2.6"

cd "${PROJECT_DIR}"
mkdir -p results/exploration_logs results/figures results/comparison_reports

opencode run \
    --model "${MODEL}" \
    --agent "psc-explorer" \
    --dangerously-skip-permissions \
    --title "PSC Single Roll" \
    --dir "${PROJECT_DIR}" \
    "摇骰子！随机选一条ML管线并执行。

## Step 1: 随机采样

\`\`\`bash
cd ${PROJECT_DIR} && python -c \"
import warnings; warnings.filterwarnings('ignore')
import json
from src.weighted_sampler import WeightedSampler
from src.registry import load_registry
reg = load_registry()
s = WeightedSampler(reg, only_implemented=True)
p = s.sample_pipeline()
layer_names = {1:'Data', 2:'Feature', 3:'Model', 4:'Eval', 5:'Screen'}
chosen = []
for layer in [1,2,3,4,5]:
    mid = p[layer]
    info = reg['methods'].get(mid, {})
    name = info.get('name', '?')
    chosen.append(f'{layer_names[layer]}={name}')
    print(f'  Layer {layer}: {mid} = {name}')
print('Pipeline: ' + ' -> '.join(chosen))
print('---CONFIG---')
print(json.dumps({str(k): v for k, v in p.items()}))
\" 2>&1 | grep -v DEPRECATION
\`\`\`

## Step 2: 执行

取 CONFIG 后执行：

\`\`\`bash
cd ${PROJECT_DIR} && python -c \"
import warnings; warnings.filterwarnings('ignore')
import json
from src.pipeline import run_pipeline
from src.registry import load_registry
reg = load_registry()
config = CONFIG_HERE
result = run_pipeline(config, reg)
print(f'Status: {result[\\\"status\\\"]}')
if result['status'] == 'success':
    print(f'R2: {result[\\\"metrics\\\"].get(\\\"r2\\\", \\\"N/A\\\"):.4f}')
    print(f'RMSE: {result[\\\"metrics\\\"].get(\\\"rmse\\\", \\\"N/A\\\"):.4f}')
    print(f'Samples: {result.get(\\\"n_samples\\\")}, Features: {result.get(\\\"n_features\\\")}')
else:
    print(f'Error: {result.get(\\\"error\\\")}')
print(f'Duration: {result[\\\"duration_sec\\\"]}s')
print('---RESULT---')
print(json.dumps(result, default=str))
\" 2>&1 | grep -v DEPRECATION
\`\`\`

## Step 3: 保存结果

\`\`\`bash
cd ${PROJECT_DIR} && python -c \"
import warnings; warnings.filterwarnings('ignore')
import json
from pathlib import Path
result = RESULT_HERE
p = Path('results/exploration_logs')
p.mkdir(parents=True, exist_ok=True)
existing = len(list(p.glob('run_*.json')))
run_idx = existing + 1
result['run_idx'] = run_idx
with open(p / f'run_{run_idx:04d}.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f'Saved: run_{run_idx:04d}.json')
\" 2>&1 | grep -v DEPRECATION
\`\`\`

记得所有Python命令开头加 import warnings; warnings.filterwarnings('ignore') 抑制RDKit警告。"
