#!/bin/bash
# ============================================================================
# PSC ML Random Pipeline Explorer — 使用 kimi CLI (Moonshot API) 进行随机管线探索
#
# 用法:
#   bash run_explore_kimi.sh           # 默认跑 10 轮
#   bash run_explore_kimi.sh 3         # 跑 3 轮
#   bash run_explore_kimi.sh 50        # 跑 50 轮
#
# 如果中断后想续跑:
#   bash run_explore_kimi.sh --resume 100
# ============================================================================

set -e

N_RUNS="${1:-10}"
RESUME_FLAG=""

if [ "$1" = "--resume" ]; then
    N_RUNS="${2:-10}"
    RESUME_FLAG="--continue"
fi

PROJECT_DIR="/share/yhm/test/AutoML_EDA/hybrid_agent_exploration"

echo "============================================"
echo "  PSC ML Random Pipeline Explorer"
echo "  Engine: kimi CLI (kimi-k2.6)"
echo "  Runs:   ${N_RUNS}"
echo "  Resume: ${RESUME_FLAG:-no}"
echo "============================================"

cd "${PROJECT_DIR}"
mkdir -p results/exploration_logs results/figures results/comparison_reports

kimi ${RESUME_FLAG} \
    --yolo \
    --work-dir "${PROJECT_DIR}" \
    --prompt "你是一个钙钛矿太阳能电池（PSC）机器学习管线探索Agent。

## 任务

执行 ${N_RUNS} 轮加权随机管线探索。每一轮随机组合5层方法（数据源→特征→模型→评估→筛选），执行完整的ML管线，记录结果。

$(if [ -n "${RESUME_FLAG}" ]; then echo "## 断点续跑模式"; echo "先检查 results/checkpoint.json，从上次中断的位置继续。跳过已完成的轮次。"; fi)

## 执行步骤

### Step 1: 环境检查

运行以下命令确认环境正常：

cd ${PROJECT_DIR} && python -c \"
import warnings; warnings.filterwarnings('ignore')
from src.registry import load_registry, get_methods_by_layer
reg = load_registry()
for layer in [1,2,3,4,5]:
    methods = get_methods_by_layer(reg, layer)
    impl = sum(1 for v in methods.values() if v.get('implemented'))
    print(f'Layer {layer}: {len(methods)} methods ({impl} implemented)')
print(f'Total: {len(reg[\"methods\"])} methods')
\"

### Step 2: 检查已有的checkpoint

cd ${PROJECT_DIR} && python -c \"
import json
try:
    with open('results/checkpoint.json') as f:
        state = json.load(f)
    print(f'Checkpoint: {state[\"completed_runs\"]}/{state[\"total_runs\"]} completed')
    print(f'Best R2: {state.get(\"best_r2\", \"N/A\")}')
except FileNotFoundError:
    print('No checkpoint, starting fresh.')
\"

### Step 3: 逐轮执行探索

对每一轮 (i = 1 to ${N_RUNS})，执行以下子步骤：

#### 3a. 随机采样一条管线

cd ${PROJECT_DIR} && python -c \"
import warnings; warnings.filterwarnings('ignore')
import json
from src.weighted_sampler import WeightedSampler
from src.registry import load_registry
reg = load_registry()
s = WeightedSampler(reg, only_implemented=True)
p = s.sample_pipeline()
layer_names = {1:'Data', 2:'Feature', 3:'Model', 4:'Eval', 5:'Screen'}
for layer in [1,2,3,4,5]:
    mid = p[layer]
    info = reg['methods'].get(mid, {})
    print(f'Layer {layer} ({layer_names[layer]}): {mid} = {info.get(\"name\", \"?\")}')
print('---CONFIG---')
print(json.dumps({str(k): v for k, v in p.items()}))
\" 2>&1 | grep -v DEPRECATION

#### 3b. 执行采样的管线

取上面 ---CONFIG--- 下面的 JSON 配置，替换到下面的 CONFIG 占位符：

cd ${PROJECT_DIR} && python -c \"
import warnings; warnings.filterwarnings('ignore')
import json, sys
from src.pipeline import run_pipeline
from src.registry import load_registry
reg = load_registry()
config = CONFIG_FROM_ABOVE
result = run_pipeline(config, reg)
status = result['status']
if status == 'success':
    print(f'SUCCESS: R2={result[\"metrics\"].get(\"r2\", \"N/A\"):.4f}')
    print(f'RMSE={result[\"metrics\"].get(\"rmse\", \"N/A\"):.4f}')
    print(f'Samples={result.get(\"n_samples\")}, Features={result.get(\"n_features\")}')
else:
    print(f'FAILED: {result.get(\"error\", \"unknown\")}')
print(f'Duration: {result[\"duration_sec\"]}s')
print('---RESULT---')
print(json.dumps(result, default=str))
\" 2>&1 | grep -v DEPRECATION

#### 3c. 保存结果和更新checkpoint

cd ${PROJECT_DIR} && python -c \"
import warnings; warnings.filterwarnings('ignore')
import json
from pathlib import Path
from src.checkpoint import load_checkpoint, create_initial_state, update_state, save_checkpoint
state = load_checkpoint('results/checkpoint.json')
if not state:
    state = create_initial_state(${N_RUNS})
result = RESULT_FROM_3B
update_state(state, result)
save_checkpoint('results/checkpoint.json', state)
run_idx = state['completed_runs']
p = Path('results/exploration_logs')
p.mkdir(parents=True, exist_ok=True)
with open(p / f'run_{run_idx:04d}.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f'Progress: {state[\"completed_runs\"]}/{state[\"total_runs\"]}')
print(f'Best R2 so far: {state.get(\"best_r2\", \"N/A\")}')
\" 2>&1 | grep -v DEPRECATION

#### 3d. 输出进度后继续下一轮

重复 Step 3a-3c 直到完成所有 ${N_RUNS} 轮。

**重要：如果某条管线失败，记录错误、保存失败结果、更新checkpoint，然后继续下一轮。绝对不要因为单条失败而停止整个探索。**

### Step 4: 探索完成后打印统计

cd ${PROJECT_DIR} && python -c \"
import warnings; warnings.filterwarnings('ignore')
from src.report import collect_results, print_summary
results = collect_results('results/exploration_logs')
print_summary(results)
\" 2>&1 | grep -v DEPRECATION

## 注意事项

1. 所有Python命令开头必须加 import warnings; warnings.filterwarnings('ignore') 来抑制RDKit警告
2. 管线执行可能需要几分钟（取决于特征类型和评估策略），这是正常的
3. 每轮执行完后必须保存结果到 exploration_logs/ 和更新 checkpoint.json
4. 如果某个Python命令失败，不要停止，继续下一轮
5. 先清理之前的旧结果：rm -f results/exploration_logs/run_*.json results/checkpoint.json"
