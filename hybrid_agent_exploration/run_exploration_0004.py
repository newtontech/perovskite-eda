import warnings
warnings.filterwarnings('ignore')

import json
import time
from pathlib import Path

from src.weighted_sampler import WeightedSampler
from src.registry import load_registry
from src.pipeline import run_pipeline

# Step 1: Sample pipeline
reg = load_registry()
s = WeightedSampler(reg, only_implemented=True, seed=456)
p = s.sample_pipeline()

for layer in [1, 2, 3, 4, 5]:
    mid = p[layer]
    info = reg['methods'].get(mid, {})
    print(f'Layer {layer}: {mid} = {info.get("name", "?")}')

print('---CONFIG---')
config = {str(k): v for k, v in p.items()}
print(json.dumps(config))

# Step 2: Run pipeline
result = run_pipeline(config, reg)
print(f'Status: {result["status"]}')
if result['status'] == 'success':
    print(f'R2={result["metrics"].get("r2", "N/A"):.4f}')
else:
    print(f'Error: {result.get("error")}')
print(f'Duration: {result["duration_sec"]}s')
print('---RESULT---')
print(json.dumps(result, default=str))

# Step 3: Save
result['run_idx'] = 4
p = Path('results/exploration_logs')
p.mkdir(parents=True, exist_ok=True)
with open(p / 'run_0004.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print('Saved run_0004.json')
