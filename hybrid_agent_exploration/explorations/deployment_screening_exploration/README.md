# Layer 5 — Deployment, Virtual Screening & Closed-Loop Exploration

> Path: `explorations/deployment_screening_exploration/`
>
> Focus: D52 Virtual Screening & D54 Closed-Loop Feedback for PSC additive discovery

## Overview

This folder demonstrates the **Layer 5 — Deployment** capabilities of the hybrid-agent pipeline:
- **D52 Virtual Screening**: Load a trained model, score a large candidate library, and rank molecules using multiple strategies.
- **D54 Closed-Loop Simulation**: Iteratively refine predictions by simulating experimental validation, adding results back to the training set, and retraining.

The script follows the full **Agent Loop** from `AGENTS.md`:

```
Planner → Retriever → Feature Agent → Model Agent → Evaluation Agent
→ Virtual Screening (D52) → Closed-Loop (D54) → Memory
```

## Files

| File | Purpose |
|------|---------|
| `explore_screening.py` | Main runnable script. Loads data, trains model, runs virtual screening, executes closed-loop simulation, saves artefacts. |
| `virtual_screener.py` | Modular virtual-screening library: candidate loading, scoring, ranking strategies, property filters. |
| `closed_loop.py` | Closed-loop simulator: train → screen → simulate experiment → retrain → track metrics. |
| `README.md` | This file. |

## Quick Start

```bash
cd explorations/deployment_screening_exploration
python explore_screening.py
```

### What it does

1. **Planner** — defines the scientific goal (maximize ΔPCE via screening + feedback).
2. **Retriever** — loads `../../data_cache.csv` (SMILES + `delta_pce`). Falls back to a synthetic dataset if real data is missing.
3. **Feature Agent** — computes **F21** RDKit basic descriptors for every molecule.
4. **Model Agent** — trains an **M31** RandomForestRegressor on the initial training set.
5. **Evaluation Agent** — reports 5-fold CV R² and hold-out test R²/RMSE.
6. **Virtual Screening (D52)** — scores the candidate library and applies lead-like property filters:
   - MolWt 100–600
   - LogP –2 to 5
   - TPSA ≤ 150
7. **Ranking Strategies** — generates top-20 tables for:
   - `top_k` — pure prediction score
   - `uncertainty_weighted` — upper-confidence-bound (UCB) exploration
   - `diverse_top_k` — greedy MaxMin diversity picker using ECFP4 Tanimoto distance
8. **Closed-Loop Simulation (D54)** — for each strategy, runs 5 iterations:
   - Select top-5 candidates
   - Simulate experimental measurement (ground truth + Gaussian noise σ=0.3)
   - Append to training set
   - Retrain model
   - Evaluate on hold-out validation set
9. **Visualise** — saves:
   - `score_distribution.png` — histogram of predicted ΔPCE across the library
   - `closed_loop_trajectory_{strategy}.png` — 4-panel trajectory per strategy
10. **Persist artefacts** — writes:
    - `screening_exploration_report.json` — structured summary
    - `ranking_{strategy}_top20.csv` — ranked candidate tables

## Outputs

All outputs are written to the same folder (`explorations/deployment_screening_exploration/`).

```
explorations/deployment_screening_exploration/
├── explore_screening.py
├── virtual_screener.py
├── closed_loop.py
├── README.md
├── screening_exploration_report.json
├── score_distribution.png
├── ranking_top_k_top20.csv
├── ranking_uncertainty_weighted_top20.csv
├── ranking_diverse_top_k_top20.csv
├── closed_loop_trajectory_top_k.png
├── closed_loop_trajectory_uncertainty_weighted.png
└── closed_loop_trajectory_diverse_top_k.png
```

## Design Notes

- **Graceful degradation**: If `data_cache.csv` is missing or contains no valid SMILES, the script automatically generates a synthetic dataset with a structured descriptor-to-target relationship so the closed loop still demonstrates learning.
- **Reusability**: `virtual_screener.py` and `closed_loop.py` are import-safe (add project root to `sys.path`) and can be used independently.
- **Determinism**: Fixed `RANDOM_STATE` for data splitting and model training; RDKit fingerprints are deterministic.
- **Scalability**: The default caps real data at 400 train + 1500 candidates to keep runtime under a few minutes. Increase `MAX_TRAIN` / `MAX_CANDIDATES` in `explore_screening.py` for larger screens.

## Dependencies

- `rdkit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

All are already listed in the project `requirements.txt`.

## Data Screening Criteria (数据筛选条件)

> 虚拟筛选和闭环实验的候选分子必须经过严格筛选，以确保合成可行性和实验可重复性。

### 虚拟筛选候选库筛选（D52）

| 条件 | 阈值 | 依据 |
|------|------|------|
| Molecular Weight | 100–600 Da | 可购买分子库的常见范围（Enamine, Sigma-Aldrich） |
| LogP | -2 to 5 | 确保在常见有机溶剂中的溶解性 |
| TPSA | ≤ 150 Å² | 口服/外用药物常见标准；过高影响薄膜渗透 |
| H-bond Donors | ≤ 5 | Lipinski规则；过多影响结晶 |
| H-bond Acceptors | ≤ 10 | Lipinski规则 |
| Rotatable Bonds | ≤ 10 | 适度柔性，保证界面适配性 |
| 合成可行性 | 排除含爆炸性基团（-N₃, -ONO₂）| 安全考虑 |
| 可购买性 | 优先Sigma-Aldrich / TCI / Enamine库存 | 确保实验可执行 |

### 闭环实验筛选（D54）

| 阶段 | 筛选条件 | 理由 |
|------|----------|------|
| **Initial Training Set** | Delta_PCE ∈ [-5%, +10%], n ≥ 500 | 足够数据训练初始模型 |
| **Top-k Selection** | 预测score > 0, uncertainty < σ₉₀ | 只选高置信度正增益候选 |
| **Experimental Validation** | 重复3次测量，剔除异常值（|z| > 2）| 减少实验噪声 |
| **Retraining** | 新样本n ≥ 10/round | 保证每轮有足够信息增量 |

### 闭环终止条件（自我判断）

| 条件 | 阈值 | 动作 |
|------|------|------|
| 连续3轮 top-10 实际ΔPCE无提升 | < 0.1% 改进 | 停止闭环，输出最佳候选 |
| 模型R²下降 | < 0.50 | 检查数据质量，可能过拟合 |
| 候选库穷尽 | 已筛选 > 90% | 扩大候选库或更换分子库 |
| 预测不确定性增大 | mean σ > 1.5% | 模型进入未知化学空间，需要更多探索 |

### 文献对标

| 指标 | 当前模拟 | 文献 | 差距 |
|------|----------|------|------|
| 虚拟筛选库大小 | 1,000 (合成) | 250,000 (PubChem, Co-PAS) | ❌ 规模不足 |
| 闭环轮数 | 10轮 | 通常3–5轮即可收敛 | ✅ 合理 |
| 每轮验证数 | 10个 | 文献12个验证实验 | ⚠️ 相当 |
| 实验验证 | 模拟 | Yang AFM: 真实实验 | ❌ 无真实验证 |

### 基线PCE在虚拟筛选中的角色（关键缺失）

> ⚠️ **当前虚拟筛选假设所有候选器件的基线PCE相同，这是严重错误的。**

**现实场景**：
- 分子A使低质量器件（PCE=8%→12%，Δ=+4%）表现优异
- 分子B使高质量器件（PCE=20%→22%，Δ=+2%）表现优异
- 如果不考虑基线PCE，分子A排名高于分子B，但对已拥有高质量器件的研究者无意义

**当前筛选的问题**：
```python
# 当前：仅用分子特征预测Delta_PCE
score = model.predict([MolWt, LogP, TPSA, ...])  # 缺少基线PCE！

# 正确：分子特征 + 目标器件的基线PCE
score = model.predict([MolWt, LogP, TPSA, ..., pce_baseline=15.0])
```

**推荐的虚拟筛选改进**：

1. **基线感知的筛选（Baseline-Aware Screening）**
   ```python
   def screen_for_baseline(candidates, pce_baseline):
       features = extract_features(candidates)
       features["pce_baseline"] = pce_baseline
       scores = model.predict(features)
       return ranked_candidates
   ```

2. **用户自定义基线PCE**
   - 研究者输入自己的器件基线PCE
   - 模型预测该分子对该基线的Delta_PCE
   - 输出"最适合您器件的添加剂Top-k"

3. **筛选报告应包含基线信息**
   ```
   Rank 1: 分子X
     - 预测Delta_PCE: +2.3% (基线PCE=15%)
     - 预测Delta_PCE: +4.1% (基线PCE=8%)
     - 不确定性: ±0.8%
   ```

### 关键缺失

- **基线PCE作为输入特征**: ❌ 当前筛选完全忽略器件基线质量
- **真实分子库连接**: 未连接PubChem/ZINC/Enamine API
- **DFT预筛选**: 无HOMO/LUMO/吸附能计算
- **实验闭环**: 无真实器件制备和测量
- **不确定性量化**: 仅有RF ensemble variance，无GPR

## References

- D52 / D54 definitions: see `../../AGENTS.md` → Layer 5 Deployment
- Agent Loop architecture: see `../../AGENTS.md` → AI Agent 自主探索架构
- Existing Layer 2 implementation: `../../features/rdkit_descriptors.py`
