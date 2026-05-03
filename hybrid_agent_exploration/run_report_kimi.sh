#!/bin/bash
# ============================================================================
# PSC ML Report Generator — 使用 kimi CLI 生成科研论文风格报告（含图表）
#
# 用法:
#   bash run_report_kimi.sh              # 生成完整报告
#   bash run_report_kimi.sh --continue   # 续接上次session
# ============================================================================

set -e

PROJECT_DIR="/share/yhm/test/AutoML_EDA/hybrid_agent_exploration"
CONTINUE_FLAG=""

if [ "$1" = "--continue" ]; then
    CONTINUE_FLAG="--continue"
fi

echo "============================================"
echo "  PSC ML Scientific Report Generator"
echo "  Engine: kimi CLI (kimi-k2.6)"
echo "============================================"

cd "${PROJECT_DIR}"
mkdir -p results/figures results/comparison_reports

kimi ${CONTINUE_FLAG} \
    --yolo \
    --work-dir "${PROJECT_DIR}" \
    --prompt "你是一个钙钛矿太阳能电池（PSC）机器学习研究的数据分析专家和科学写作专家。

## 任务

根据 results/exploration_logs/ 中的所有探索结果，生成8张高质量数据可视化图表和一份符合SCI科研论文结构的完整报告。

## Part A: 生成图表

先创建一个Python脚本 generate_figures.py 一次性生成所有8张图：

cd ${PROJECT_DIR}

创建 generate_figures.py 文件，内容如下：

import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path
from src.report import collect_results

plt.rcParams.update({'font.size': 11, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
Path('results/figures').mkdir(parents=True, exist_ok=True)

results = collect_results('results/exploration_logs')
successful = [r for r in results if r.get('status') == 'success']
failed = [r for r in results if r.get('status') == 'error']

if not successful:
    print('No successful runs found. Cannot generate figures.')
    exit(0)

r2_values = [r['metrics'].get('r2', 0) for r in successful]

# === Figure 1: R2 Distribution ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(r2_values, bins=min(20, len(r2_values)//2+1), color='#2196F3', edgecolor='white', alpha=0.8)
axes[0].axvline(x=np.mean(r2_values), color='red', linestyle='--', label=f'Mean R²={np.mean(r2_values):.4f}')
axes[0].axvline(x=max(r2_values), color='green', linestyle='--', label=f'Best R²={max(r2_values):.4f}')
axes[0].set_xlabel('R² Score'); axes[0].set_ylabel('Count')
axes[0].set_title('(a) Distribution of R² Scores'); axes[0].legend()

sorted_r2 = sorted(r2_values, reverse=True)
axes[1].plot(range(1, len(sorted_r2)+1), sorted_r2, 'o-', color='#FF5722', markersize=4)
axes[1].axhline(y=0.1624, color='gray', linestyle=':', label='AutoGluon baseline R²=0.1624')
axes[1].set_xlabel('Pipeline Rank'); axes[1].set_ylabel('R² Score')
axes[1].set_title('(b) R² Score vs. Pipeline Rank'); axes[1].legend()
plt.tight_layout(); plt.savefig('results/figures/fig1_r2_distribution.png'); plt.close()
print('Saved fig1')

# === Figure 2: Feature Comparison ===
feat_r2 = defaultdict(list)
for r in successful:
    cfg = r.get('pipeline_config', {})
    mid = cfg.get('2') or cfg.get(2)
    if mid: feat_r2[mid].append(r['metrics'].get('r2', 0))

if feat_r2:
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = sorted(feat_r2.keys(), key=lambda k: np.mean(feat_r2[k]), reverse=True)
    data = [feat_r2[k] for k in labels]
    short_labels = [l.replace('F22_','').replace('F21_','').replace('_',' ').title() for l in labels]
    bp = ax.boxplot(data, labels=short_labels, patch_artist=True, showmeans=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color)
    ax.set_ylabel('R² Score'); ax.set_title('Feature Representation Methods Comparison')
    ax.tick_params(axis='x', rotation=30, labelsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig('results/figures/fig2_feature_comparison.png'); plt.close()
    print('Saved fig2')

# === Figure 3: Model Comparison ===
model_r2 = defaultdict(list)
for r in successful:
    cfg = r.get('pipeline_config', {})
    mid = cfg.get('3') or cfg.get(3)
    if mid: model_r2[mid].append(r['metrics'].get('r2', 0))

if model_r2:
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = sorted(model_r2.keys(), key=lambda k: np.mean(model_r2[k]), reverse=True)
    data = [model_r2[k] for k in labels]
    short_labels = [l.replace('M31_','').replace('_',' ').title() for l in labels]
    bp = ax.boxplot(data, labels=short_labels, patch_artist=True, showmeans=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color)
    ax.set_ylabel('R² Score'); ax.set_title('Machine Learning Models Comparison')
    ax.tick_params(axis='x', rotation=30, labelsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig('results/figures/fig3_model_comparison.png'); plt.close()
    print('Saved fig3')

# === Figure 4: Eval Strategy Comparison ===
eval_r2 = defaultdict(list)
for r in successful:
    cfg = r.get('pipeline_config', {})
    mid = cfg.get('4') or cfg.get(4)
    if mid: eval_r2[mid].append(r['metrics'].get('r2', 0))

if eval_r2:
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = sorted(eval_r2.keys(), key=lambda k: np.mean(eval_r2[k]), reverse=True)
    data = [eval_r2[k] for k in labels]
    short_labels = [l.replace('E42_','').replace('E43_','').replace('E44_','').replace('E45_','').replace('_',' ').title() for l in labels]
    bp = ax.boxplot(data, labels=short_labels, patch_artist=True, showmeans=True)
    colors = plt.cm.Paired(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color)
    ax.set_ylabel('R² Score'); ax.set_title('Evaluation Strategy Comparison')
    ax.tick_params(axis='x', rotation=30, labelsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig('results/figures/fig4_eval_comparison.png'); plt.close()
    print('Saved fig4')

# === Figure 5: Top 10 Pipelines ===
top10 = sorted(successful, key=lambda r: r['metrics'].get('r2', 0), reverse=True)[:10]
fig, ax = plt.subplots(figsize=(14, 6))
y_labels = []; r2_top = []
for i, r in enumerate(top10):
    cfg = r.get('pipeline_config', {})
    r2_top.append(r['metrics'].get('r2', 0))
    feat = (cfg.get('2') or cfg.get(2, '?')).replace('F22_','').replace('F21_','')
    model = (cfg.get('3') or cfg.get(3, '?')).replace('M31_','')
    eval_m = (cfg.get('4') or cfg.get(4, '?')).replace('E42_','').replace('E43_','').replace('E44_','').replace('E45_','')
    y_labels.append(f'#{i+1}: {feat} + {model} + {eval_m}')

colors_bar = plt.cm.RdYlGn(np.array([(v - min(r2_top)) / (max(r2_top) - min(r2_top) + 1e-8) for v in r2_top]))
bars = ax.barh(range(len(r2_top)), r2_top, color=colors_bar, edgecolor='gray', linewidth=0.5)
ax.set_yticks(range(len(y_labels))); ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel('R² Score'); ax.set_title('Top 10 Pipeline Configurations'); ax.invert_yaxis()
for bar, r2 in zip(bars, r2_top):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f'{r2:.4f}', va='center', fontsize=9)
plt.tight_layout(); plt.savefig('results/figures/fig5_top10_pipelines.png'); plt.close()
print('Saved fig5')

# === Figure 6: Feature x Model Heatmap ===
pair_r2 = defaultdict(list)
for r in successful:
    cfg = r.get('pipeline_config', {})
    feat = cfg.get('2') or cfg.get(2); model = cfg.get('3') or cfg.get(3)
    if feat and model: pair_r2[(feat, model)].append(r['metrics'].get('r2', 0))

feats = sorted(set(k[0] for k in pair_r2.keys()))
models = sorted(set(k[1] for k in pair_r2.keys()))
matrix = np.full((len(feats), len(models)), np.nan)
for i, f in enumerate(feats):
    for j, m in enumerate(models):
        vals = pair_r2.get((f, m), [])
        if vals: matrix[i, j] = np.mean(vals)

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(models))); ax.set_yticks(range(len(feats)))
ax.set_xticklabels([m.replace('M31_','') for m in models], rotation=45, ha='right', fontsize=9)
ax.set_yticklabels([f.replace('F22_','').replace('F21_','') for f in feats], fontsize=9)
for i in range(len(feats)):
    for j in range(len(models)):
        if not np.isnan(matrix[i,j]):
            ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', fontsize=7, color='white' if matrix[i,j]<0 else 'black')
plt.colorbar(im, label='Mean R²')
ax.set_title('Feature × Model Interaction Matrix (Mean R²)'); ax.set_xlabel('Model'); ax.set_ylabel('Feature')
plt.tight_layout(); plt.savefig('results/figures/fig6_feature_model_heatmap.png'); plt.close()
print('Saved fig6')

# === Figure 7: Convergence ===
successful_sorted = sorted(successful, key=lambda r: r.get('run_idx', 0))
r2_seq = [r['metrics'].get('r2', 0) for r in successful_sorted]
cum_best = np.maximum.accumulate(r2_seq)
runs = range(1, len(r2_seq)+1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(runs, r2_seq, 'o', color='#90CAF9', markersize=4, alpha=0.6, label='Individual R²')
ax.plot(runs, cum_best, '-', color='#F44336', linewidth=2, label='Best R² so far')
ax.axhline(y=0.1624, color='gray', linestyle=':', linewidth=1, label='AutoGluon baseline (R²=0.1624)')
ax.fill_between(runs, r2_seq, cum_best, alpha=0.1, color='red')
ax.set_xlabel('Exploration Run'); ax.set_ylabel('R² Score')
ax.set_title('Exploration Convergence: Best R² Over Time'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('results/figures/fig7_convergence.png'); plt.close()
print('Saved fig7')

# === Figure 8: Top 20 Composition ===
top20 = sorted(successful, key=lambda r: r['metrics'].get('r2', 0), reverse=True)[:20]
layer_colors = {2: '#2196F3', 3: '#FF9800', 4: '#4CAF50'}
layer_names = {2: 'Feature', 3: 'Model', 4: 'Eval'}

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
for idx, layer in enumerate([2, 3, 4]):
    methods = []
    for r in top20:
        cfg = r.get('pipeline_config', {})
        mid = cfg.get(layer) or cfg.get(str(layer), '?')
        methods.append(mid)
    counter = Counter(methods)
    labels = [k.replace('F22_','').replace('F21_','').replace('M31_','').replace('E42_','').replace('E43_','').replace('E44_','').replace('E45_','').replace('_',' ') for k in counter.keys()]
    counts = list(counter.values())
    axes[idx].barh(labels, counts, color=layer_colors[layer], alpha=0.8, edgecolor='white')
    axes[idx].set_title(f'{layer_names[layer]} Method Frequency in Top 20 Pipelines', fontsize=11)
    axes[idx].set_xlabel('Count')
plt.tight_layout(); plt.savefig('results/figures/fig8_top20_composition.png'); plt.close()
print('Saved fig8')

print(f'All 8 figures saved to results/figures/')
print(f'Total: {len(results)} runs, {len(successful)} success, {len(failed)} failed')
print(f'R2 range: {min(r2_values):.4f} - {max(r2_values):.4f}, mean: {np.mean(r2_values):.4f}')

然后用 bash 执行：cd ${PROJECT_DIR} && python generate_figures.py

## Part B: 收集统计数据

cd ${PROJECT_DIR} && python -c "
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np
from collections import defaultdict
from src.report import collect_results
results = collect_results('results/exploration_logs')
successful = [r for r in results if r.get('status') == 'success']
failed = [r for r in results if r.get('status') == 'error']
print('=== SUMMARY STATISTICS ===')
print(f'Total: {len(results)}, Success: {len(successful)}, Failed: {len(failed)}')
if successful:
    r2_vals = [r['metrics'].get('r2', 0) for r in successful]
    rmse_vals = [r['metrics'].get('rmse', 0) for r in successful if r['metrics'].get('rmse')]
    dur_vals = [r.get('duration_sec', 0) for r in successful]
    print(f'R2: min={min(r2_vals):.4f}, max={max(r2_vals):.4f}, mean={np.mean(r2_vals):.4f}, std={np.std(r2_vals):.4f}')
    if rmse_vals: print(f'RMSE: min={min(rmse_vals):.4f}, max={max(rmse_vals):.4f}, mean={np.mean(rmse_vals):.4f}')
    print(f'Duration: mean={np.mean(dur_vals):.1f}s, total={sum(dur_vals):.1f}s')
    best = max(successful, key=lambda r: r['metrics'].get('r2', 0))
    cfg = best.get('pipeline_config', {})
    print(f'Best R2: {best[\"metrics\"].get(\"r2\", 0):.4f}')
    for layer in [1,2,3,4,5]:
        mid = cfg.get(layer) or cfg.get(str(layer), '?')
        print(f'  Layer {layer}: {mid}')
    print('=== PER-FEATURE ===')
    feat_r2 = defaultdict(list)
    for r in successful:
        cfg2 = r.get('pipeline_config', {})
        mid = cfg2.get('2') or cfg2.get(2)
        if mid: feat_r2[mid].append(r['metrics'].get('r2', 0))
    for k in sorted(feat_r2.keys(), key=lambda x: np.mean(feat_r2[x]), reverse=True):
        vals = feat_r2[k]
        print(f'  {k}: mean={np.mean(vals):.4f}, max={max(vals):.4f}, n={len(vals)}')
    print('=== PER-MODEL ===')
    model_r2 = defaultdict(list)
    for r in successful:
        cfg3 = r.get('pipeline_config', {})
        mid = cfg3.get('3') or cfg3.get(3)
        if mid: model_r2[mid].append(r['metrics'].get('r2', 0))
    for k in sorted(model_r2.keys(), key=lambda x: np.mean(model_r2[x]), reverse=True):
        vals = model_r2[k]
        print(f'  {k}: mean={np.mean(vals):.4f}, max={max(vals):.4f}, n={len(vals)}')
" 2>&1 | grep -v DEPRECATION

## Part C: 撰写科研报告

根据以上所有数据和图表，撰写完整的科研报告，保存到 results/comparison_reports/scientific_report.md。

报告必须严格遵循以下结构（英文撰写）：

# Machine Learning Pipeline Exploration for Perovskite Solar Cell Additive Design: A Random Search Approach

## Abstract
(200-300 words: background, methods, key findings, conclusions)

## 1. Introduction
- 1.1 Perovskite Solar Cells and Molecular Additives
- 1.2 Machine Learning in Materials Science
- 1.3 Random Search vs. Grid Search for Pipeline Optimization
- 1.4 Objectives of This Study

## 2. Methods
- 2.1 Dataset Description
- 2.2 Five-Layer Pipeline Architecture
- 2.3 Feature Representations
- 2.4 Machine Learning Models
- 2.5 Evaluation Strategies
- 2.6 Weighted Random Sampling

## 3. Results

用 markdown 图片引用（相对路径）嵌入所有8张图，每张图配详细的 caption：
![Figure 1](../figures/fig1_r2_distribution.png)
**Figure 1.** ...

![Figure 2](../figures/fig2_feature_comparison.png)
**Figure 2.** ...

(对 fig3-fig8 同理)

### 3.1 Overall Exploration Statistics
### 3.2 Feature Representation Analysis
### 3.3 Model Performance Comparison
### 3.4 Feature-Model Interaction
### 3.5 Evaluation Strategy Impact
### 3.6 Convergence Analysis

## 4. Discussion
- 4.1 Key Findings
- 4.2 Limitations
- 4.3 Comparison with AutoGluon Baseline (R²=0.1624)
- 4.4 Recommendations for Future Work

## 5. Conclusions

## References

**重要：**
1. 英文撰写，符合SCI论文标准
2. 统计数据必须来自实际结果，不要编造
3. 图片相对路径引用：![Figure X](../figures/figX_xxx.png)
4. 结合PSC领域的专业知识讨论
5. 全文保存到 results/comparison_reports/scientific_report.md"
