#!/bin/bash
# ============================================================================
# PSC ML Report Generator — 使用 OpenCode + Kimi K2.6 生成科研论文风格报告
#
# 用法:
#   bash run_report.sh              # 生成完整报告（含图片）
#   bash run_report.sh --continue   # 续接上次session继续生成
#
# 输出:
#   results/figures/                # 所有图片 (300 DPI PNG)
#   results/comparison_reports/scientific_report.md   # 科研论文结构报告
# ============================================================================

set -e

PROJECT_DIR="/share/yhm/test/AutoML_EDA/hybrid_agent_exploration"
MODEL="openrouter/moonshotai/kimi-k2.6"
AGENT="psc-explorer"
CONTINUE_FLAG=""

if [ "$1" = "--continue" ]; then
    CONTINUE_FLAG="--continue"
fi

echo "============================================"
echo "  PSC ML Scientific Report Generator"
echo "  Model:  ${MODEL}"
echo "============================================"

cd "${PROJECT_DIR}"
mkdir -p results/figures results/comparison_reports

opencode run ${CONTINUE_FLAG} \
    --model "${MODEL}" \
    --agent "${AGENT}" \
    --dangerously-skip-permissions \
    --title "PSC Scientific Report Generation" \
    --dir "${PROJECT_DIR}" \
    "你是一个钙钛矿太阳能电池（PSC）机器学习研究的数据分析专家和科学写作专家。

## 任务

根据 results/exploration_logs/ 中的所有探索结果，生成一份符合SCI科研论文结构的完整报告，包含高质量数据可视化图表。

## Step 1: 收集和汇总数据

\`\`\`bash
cd ${PROJECT_DIR} && python -c \"
import warnings; warnings.filterwarnings('ignore')
import json
from pathlib import Path
from src.report import collect_results
results = collect_results('results/exploration_logs')
successful = [r for r in results if r.get('status') == 'success']
failed = [r for r in results if r.get('status') == 'error']
print(f'Total runs: {len(results)}')
print(f'Successful: {len(successful)}')
print(f'Failed: {len(failed)}')
if successful:
    r2_values = [r['metrics'].get('r2', 0) for r in successful]
    print(f'R2 range: {min(r2_values):.4f} - {max(r2_values):.4f}')
    print(f'R2 mean: {sum(r2_values)/len(r2_values):.4f}')
    best = max(successful, key=lambda r: r['metrics'].get('r2', 0))
    print(f'Best R2: {best[\\\"metrics\\\"].get(\\\"r2\\\", 0):.4f}')
    print(f'Best pipeline: {best.get(\\\"pipeline_config\\\", {})}')
\"
\`\`\`

## Step 2: 生成所有数据可视化图表

对每个图表，创建独立的Python脚本并执行。所有图片保存到 results/figures/，分辨率300 DPI。

### 图表1: 探索总览 — R2分布直方图

\`\`\`bash
cd ${PROJECT_DIR} && python << 'PYEOF'
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from src.report import collect_results

results = collect_results('results/exploration_logs')
successful = [r for r in results if r.get('status') == 'success']

if successful:
    r2_values = [r['metrics'].get('r2', 0) for r in successful]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R2 histogram
    axes[0].hist(r2_values, bins=20, color='#2196F3', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=np.mean(r2_values), color='red', linestyle='--', label=f'Mean R²={np.mean(r2_values):.4f}')
    axes[0].axvline(x=max(r2_values), color='green', linestyle='--', label=f'Best R²={max(r2_values):.4f}')
    axes[0].set_xlabel('R² Score', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('(a) Distribution of R² Scores', fontsize=13)
    axes[0].legend(fontsize=10)

    # R2 cumulative
    sorted_r2 = sorted(r2_values, reverse=True)
    axes[1].plot(range(1, len(sorted_r2)+1), sorted_r2, 'o-', color='#FF5722', markersize=4)
    axes[1].axhline(y=0.1624, color='gray', linestyle=':', label='AutoGluon baseline R²=0.1624')
    axes[1].set_xlabel('Pipeline Rank', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('(b) R² Score vs. Pipeline Rank', fontsize=13)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('results/figures/fig1_r2_distribution.png', dpi=300, bbox_inches='tight')
    print('Saved: results/figures/fig1_r2_distribution.png')
else:
    print('No successful runs to plot.')
PYEOF
\`\`\`

### 图表2: 特征表示方法对比 — 箱线图

\`\`\`bash
cd ${PROJECT_DIR} && python << 'PYEOF'
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from src.report import collect_results

results = collect_results('results/exploration_logs')
successful = [r for r in results if r.get('status') == 'success')

if successful:
    feat_r2 = defaultdict(list)
    for r in successful:
        cfg = r.get('pipeline_config', {})
        mid = cfg.get('2') or cfg.get(2)
        if mid:
            feat_r2[mid].append(r['metrics'].get('r2', 0))

    if feat_r2:
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = sorted(feat_r2.keys(), key=lambda k: np.mean(feat_r2[k]), reverse=True)
        data = [feat_r2[k] for k in labels]
        short_labels = [l.replace('F22_', '').replace('F21_', '').replace('_', ' ').title() for l in labels]

        bp = ax.boxplot(data, labels=short_labels, patch_artist=True, showmeans=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Feature Representation Methods Comparison', fontsize=13)
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/figures/fig2_feature_comparison.png', dpi=300, bbox_inches='tight')
        print('Saved: results/figures/fig2_feature_comparison.png')
PYEOF
\`\`\`

### 图表3: ML模型对比 — 箱线图

\`\`\`bash
cd ${PROJECT_DIR} && python << 'PYEOF'
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from src.report import collect_results

results = collect_results('results/exploration_logs')
successful = [r for r in results if r.get('status') == 'success']

if successful:
    model_r2 = defaultdict(list)
    for r in successful:
        cfg = r.get('pipeline_config', {})
        mid = cfg.get('3') or cfg.get(3)
        if mid:
            model_r2[mid].append(r['metrics'].get('r2', 0))

    if model_r2:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = sorted(model_r2.keys(), key=lambda k: np.mean(model_r2[k]), reverse=True)
        data = [model_r2[k] for k in labels]
        short_labels = [l.replace('M31_', '').replace('_', ' ').title() for l in labels]

        bp = ax.boxplot(data, labels=short_labels, patch_artist=True, showmeans=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Machine Learning Models Comparison', fontsize=13)
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/figures/fig3_model_comparison.png', dpi=300, bbox_inches='tight')
        print('Saved: results/figures/fig3_model_comparison.png')
PYEOF
\`\`\`

### 图表4: 评估策略对比

\`\`\`bash
cd ${PROJECT_DIR} && python << 'PYEOF'
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from src.report import collect_results

results = collect_results('results/exploration_logs')
successful = [r for r in results if r.get('status') == 'success']

if successful:
    eval_r2 = defaultdict(list)
    for r in successful:
        cfg = r.get('pipeline_config', {})
        mid = cfg.get('4') or cfg.get(4)
        if mid:
            eval_r2[mid].append(r['metrics'].get('r2', 0))

    if eval_r2:
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = sorted(eval_r2.keys(), key=lambda k: np.mean(eval_r2[k]), reverse=True)
        data = [eval_r2[k] for k in labels]
        short_labels = [l.replace('E42_', '').replace('E43_', '').replace('E44_', '').replace('E45_', '').replace('_', ' ').title() for l in labels]

        bp = ax.boxplot(data, labels=short_labels, patch_artist=True, showmeans=True)
        colors = plt.cm.Paired(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Evaluation Strategy Comparison', fontsize=13)
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/figures/fig4_eval_comparison.png', dpi=300, bbox_inches='tight')
        print('Saved: results/figures/fig4_eval_comparison.png')
PYEOF
\`\`\`

### 图表5: Top 10 管线组合热力图

\`\`\`bash
cd ${PROJECT_DIR} && python << 'PYEOF'
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.report import collect_results

results = collect_results('results/exploration_logs')
successful = sorted([r for r in results if r.get('status') == 'success'],
                    key=lambda r: r['metrics'].get('r2', 0), reverse=True)

if successful:
    top10 = successful[:10]
    fig, ax = plt.subplots(figsize=(14, 6))

    layer_names = {2: 'Feature', 3: 'Model', 4: 'Evaluation'}
    y_labels = []
    r2_values = []
    feat_colors = {}
    model_colors = {}
    eval_colors = {}

    feat_cm = plt.cm.Set3
    model_cm = plt.cm.tab20
    eval_cm = plt.cm.Paired

    for i, r in enumerate(top10):
        cfg = r.get('pipeline_config', {})
        r2 = r['metrics'].get('r2', 0)
        r2_values.append(r2)
        feat = (cfg.get('2') or cfg.get(2, '?')).replace('F22_', '').replace('F21_', '')
        model = (cfg.get('3') or cfg.get(3, '?')).replace('M31_', '')
        eval_m = (cfg.get('4') or cfg.get(4, '?')).replace('E42_', '').replace('E43_', '').replace('E44_', '').replace('E45_', '')
        y_labels.append(f'#{i+1}: {feat} + {model} + {eval_m}')

    colors_bar = plt.cm.RdYlGn(np.array([(v - min(r2_values)) / (max(r2_values) - min(r2_values) + 1e-8) for v in r2_values]))

    bars = ax.barh(range(len(r2_values)), r2_values, color=colors_bar, edgecolor='gray', linewidth=0.5)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('R² Score', fontsize=12)
    ax.set_title('Top 10 Pipeline Configurations', fontsize=13)
    ax.invert_yaxis()

    for bar, r2 in zip(bars, r2_values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{r2:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/figures/fig5_top10_pipelines.png', dpi=300, bbox_inches='tight')
    print('Saved: results/figures/fig5_top10_pipelines.png')
PYEOF
\`\`\`

### 图表6: 特征-模型交互热力图

\`\`\`bash
cd ${PROJECT_DIR} && python << 'PYEOF'
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from src.report import collect_results

results = collect_results('results/exploration_logs')
successful = [r for r in results if r.get('status') == 'success']

if successful:
    pair_r2 = defaultdict(list)
    for r in successful:
        cfg = r.get('pipeline_config', {})
        feat = cfg.get('2') or cfg.get(2)
        model = cfg.get('3') or cfg.get(3)
        if feat and model:
            pair_r2[(feat, model)].append(r['metrics'].get('r2', 0))

    feats = sorted(set(k[0] for k in pair_r2.keys()))
    models = sorted(set(k[1] for k in pair_r2.keys()))

    matrix = np.full((len(feats), len(models)), np.nan)
    for i, f in enumerate(feats):
        for j, m in enumerate(models):
            vals = pair_r2.get((f, m), [])
            if vals:
                matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels([m.replace('M31_', '') for m in models], rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([f.replace('F22_', '').replace('F21_', '') for f in feats], fontsize=9)

    for i in range(len(feats)):
        for j in range(len(models)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', fontsize=7,
                       color='white' if matrix[i, j] < 0 else 'black')

    plt.colorbar(im, label='Mean R²')
    ax.set_title('Feature × Model Interaction Matrix (Mean R²)', fontsize=13)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    plt.tight_layout()
    plt.savefig('results/figures/fig6_feature_model_heatmap.png', dpi=300, bbox_inches='tight')
    print('Saved: results/figures/fig6_feature_model_heatmap.png')
PYEOF
\`\`\`

### 图表7: 探索收敛曲线 — 随着运行次数增加Best R2的变化

\`\`\`bash
cd ${PROJECT_DIR} && python << 'PYEOF'
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.report import collect_results

results = collect_results('results/exploration_logs')
successful = sorted([r for r in results if r.get('status') == 'success'],
                    key=lambda r: r.get('run_idx', 0))

if successful:
    r2_values = [r['metrics'].get('r2', 0) for r in successful]
    cumulative_best = np.maximum.accumulate(r2_values)
    runs = range(1, len(r2_values) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(runs, r2_values, 'o', color='#90CAF9', markersize=4, alpha=0.6, label='Individual R²')
    ax.plot(runs, cumulative_best, '-', color='#F44336', linewidth=2, label='Best R² so far')
    ax.axhline(y=0.1624, color='gray', linestyle=':', linewidth=1, label='AutoGluon baseline (R²=0.1624)')
    ax.fill_between(runs, r2_values, cumulative_best, alpha=0.1, color='red')

    ax.set_xlabel('Exploration Run', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Exploration Convergence: Best R² Over Time', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/fig7_convergence.png', dpi=300, bbox_inches='tight')
    print('Saved: results/figures/fig7_convergence.png')
PYEOF
\`\`\`

### 图表8: 管线组合桑基图（或堆叠条形图）

\`\`\`bash
cd ${PROJECT_DIR} && python << 'PYEOF'
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from src.report import collect_results

results = collect_results('results/exploration_logs')
successful = sorted([r for r in results if r.get('status') == 'success'],
                    key=lambda r: r['metrics'].get('r2', 0), reverse=True)

if successful:
    top20 = successful[:20]
    layer_colors = {2: '#2196F3', 3: '#FF9800', 4: '#4CAF50'}
    layer_names = {2: 'Feature', 3: 'Model', 4: 'Eval'}

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    for idx, layer in enumerate([2, 3, 4]):
        methods = []
        for r in top20:
            cfg = r.get('pipeline_config', {})
            mid = cfg.get(layer) or cfg.get(str(layer), '?')
            methods.append(mid)

        counter = Counter(methods)
        labels = [k.replace('F22_', '').replace('F21_', '').replace('M31_', '').replace('E42_', '').replace('E43_', '').replace('E44_', '').replace('E45_', '').replace('_', ' ') for k in counter.keys()]
        counts = list(counter.values())

        axes[idx].barh(labels, counts, color=layer_colors[layer], alpha=0.8, edgecolor='white')
        axes[idx].set_title(f'{layer_names[layer]} Method Frequency in Top 20 Pipelines', fontsize=11)
        axes[idx].set_xlabel('Count', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/figures/fig8_top20_composition.png', dpi=300, bbox_inches='tight')
    print('Saved: results/figures/fig8_top20_composition.png')
PYEOF
\`\`\`

## Step 3: 收集统计摘要数据

\`\`\`bash
cd ${PROJECT_DIR} && python << 'PYEOF'
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
    if rmse_vals:
        print(f'RMSE: min={min(rmse_vals):.4f}, max={max(rmse_vals):.4f}, mean={np.mean(rmse_vals):.4f}')
    print(f'Duration: mean={np.mean(dur_vals):.1f}s, total={sum(dur_vals):.1f}s')

    print()
    print('=== BEST PIPELINE ===')
    best = max(successful, key=lambda r: r['metrics'].get('r2', 0))
    cfg = best.get('pipeline_config', {})
    print(f'R2: {best[\"metrics\"].get(\"r2\", 0):.4f}')
    print(f'RMSE: {best[\"metrics\"].get(\"rmse\", \"N/A\")}')
    for layer in [1,2,3,4,5]:
        mid = cfg.get(layer) or cfg.get(str(layer), '?')
        print(f'  Layer {layer}: {mid}')

    print()
    print('=== PER-FEATURE STATS ===')
    feat_r2 = defaultdict(list)
    for r in successful:
        cfg2 = r.get('pipeline_config', {})
        mid = cfg2.get('2') or cfg2.get(2)
        if mid:
            feat_r2[mid].append(r['metrics'].get('r2', 0))
    for k in sorted(feat_r2.keys(), key=lambda x: np.mean(feat_r2[x]), reverse=True):
        vals = feat_r2[k]
        print(f'  {k}: mean={np.mean(vals):.4f}, max={max(vals):.4f}, n={len(vals)}')

    print()
    print('=== PER-MODEL STATS ===')
    model_r2 = defaultdict(list)
    for r in successful:
        cfg3 = r.get('pipeline_config', {})
        mid = cfg3.get('3') or cfg3.get(3)
        if mid:
            model_r2[mid].append(r['metrics'].get('r2', 0))
    for k in sorted(model_r2.keys(), key=lambda x: np.mean(model_r2[x]), reverse=True):
        vals = model_r2[k]
        print(f'  {k}: mean={np.mean(vals):.4f}, max={max(vals):.4f}, n={len(vals)}')
PYEOF
\`\`\`

## Step 4: 撰写科研论文风格报告

根据以上所有数据和图表，撰写一份完整的科研报告，保存到 \`results/comparison_reports/scientific_report.md\`。

报告必须严格遵循以下结构：

---

# Machine Learning Pipeline Exploration for Perovskite Solar Cell Additive Design: A Random Search Approach

## Abstract
（200-300字摘要：背景、方法、关键发现、结论）

## 1. Introduction
- 1.1 Perovskite Solar Cells and Molecular Additives
- 1.2 Machine Learning in Materials Science
- 1.3 Random Search vs. Grid Search for Pipeline Optimization
- 1.4 Objectives of This Study

## 2. Methods
- 2.1 Dataset Description (来源、样本数、特征、目标变量 Delta_PCE)
- 2.2 Five-Layer Pipeline Architecture (数据源→特征→模型→评估→筛选，每层的方法列表和加权采样策略)
- 2.3 Feature Representations (ECFP4/6, MACCS, KRFP, Atom Pair, Topological Torsion, RDKit Descriptors 的原理和参数)
- 2.4 Machine Learning Models (RF, XGBoost, LightGBM, CatBoost, GB, SVR, KNN, ElasticNet, Ridge, Lasso 的超参数设置)
- 2.5 Evaluation Strategies (Random Split, 5-fold/10-fold CV, Optuna, Grid Search, SHAP)
- 2.6 Weighted Random Sampling (权重公式：base × novelty × availability × PSC_verification)
- 2.7 Statistical Analysis

## 3. Results
用文字描述每个图的关键发现，并引用对应的图：

![Figure 1](../figures/fig1_r2_distribution.png)
**Figure 1.** (a) Distribution of R² scores across all successful pipeline runs. (b) R² score vs. pipeline rank showing the performance decay.

![Figure 2](../figures/fig2_feature_comparison.png)
**Figure 2.** Comparison of R² scores across different molecular feature representations.

![Figure 3](../figures/fig3_model_comparison.png)
**Figure 3.** Comparison of R² scores across different machine learning models.

![Figure 4](../figures/fig4_eval_comparison.png)
**Figure 4.** Impact of evaluation strategy on reported R² scores.

![Figure 5](../figures/fig5_top10_pipelines.png)
**Figure 5.** Top 10 performing pipeline configurations ranked by R² score.

![Figure 6](../figures/fig6_feature_model_heatmap.png)
**Figure 6.** Feature × Model interaction matrix showing mean R² for each combination.

![Figure 7](../figures/fig7_convergence.png)
**Figure 7.** Exploration convergence: best R² achieved over successive random trials.

![Figure 8](../figures/fig8_top20_composition.png)
**Figure 8.** Composition of the top 20 pipelines by feature, model, and evaluation method.

### 3.1 Overall Exploration Statistics
（总运行数、成功率、R2范围、与AutoGluon基线对比）

### 3.2 Feature Representation Analysis
（哪种特征表示效果最好？为什么？结合化学信息学知识解释）

### 3.3 Model Performance Comparison
（哪个模型最好？集成方法 vs. 线性方法 vs. 近邻方法的表现差异）

### 3.4 Feature-Model Interaction
（哪些特征-模型组合最有效？热力图的解读）

### 3.5 Evaluation Strategy Impact
（不同评估策略对R2的影响，过拟合风险分析）

### 3.6 Convergence Analysis
（需要多少次随机探索才能找到好的管线？效率分析）

## 4. Discussion
- 4.1 Key Findings and Implications for PSC Research
- 4.2 Limitations of Random Search
- 4.3 Comparison with AutoGluon Baseline
- 4.4 Recommendations for Future Work (深度学习模型、图神经网络、迁移学习)

## 5. Conclusions
（主要结论3-5条）

## References
（引用相关文献：PSC ML综述、ECFP原始论文、XGBoost论文等）

---

**重要要求：**
1. 报告必须用英文撰写（符合SCI论文标准）
2. 每个图表必须有详细的图注（caption）
3. 图片用相对路径引用：\`![Figure X](../figures/figX_xxx.png)\`
4. 讨论部分需要结合钙钛矿太阳能电池领域的专业知识
5. 统计数据必须来自实际的探索结果，不要编造
6. 如果某些图表因为没有成功运行而缺失，跳过该图，标注'Data not available'
7. 报告全文保存到 results/comparison_reports/scientific_report.md"
