# Evaluation Strategy Exploration (Layer 4)

This directory contains an autonomous exploration of **evaluation, validation, and optimisation strategies** (Layer 4 of the Hybrid Agent Exploration architecture).

## Scope

| Code | Layer 4 Component | Status |
|------|-------------------|--------|
| E42 | Data splitting (random, scaffold, temporal) | ✅ Implemented |
| E43 | Cross-validation (k-fold, repeated, nested) | ✅ Implemented |
| E45 | Metrics & uncertainty (R², RMSE, MAE, ensemble variance, bootstrap CI) | ✅ Implemented |

## Files

| File | Description |
|------|-------------|
| `explore_evaluation.py` | Main script: generates data, runs all splits / CV strategies, trains models, compares results |
| `splitters.py` | Splitting implementations: `RandomSplitter`, `ScaffoldSplitter`, `TemporalSplitter`, `KFoldSplitter`, `RepeatedKFoldSplitter`, `NestedCVSplitter`, `ScaffoldKFoldSplitter` |
| `metrics.py` | Metrics & uncertainty: `rmse`, `mae`, `r2`, `EnsembleUncertainty`, `BootstrapUncertainty`, `aggregate_cv_results` |
| `README.md` | This file |

## Quick Start

```bash
cd explorations/evaluation_strategy_exploration
python explore_evaluation.py
```

### Requirements

All dependencies are listed in the project `requirements.txt`:

- `scikit-learn >= 1.3.0`
- `rdkit-pypi >= 2023.03.1`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`

## What the Script Does

1. **Data Generation**  
   Creates ~1,200 synthetic molecules with SMILES strings, RDKit descriptors, Morgan fingerprints, and a noisy regression target. The true target is a known sparse linear combination of key descriptors (MolWt, LogP, H-donors, H-acceptors, rotatable bonds, TPSA). A `year` column (2015–2025) enables temporal splitting.

2. **Simple Splits (E42)**
   - **Random split** – baseline; molecules are assigned randomly.
   - **Scaffold split** – Bemis-Murcko scaffolds are clustered; no scaffold appears in both train and test. This is a stricter test of generalisation for molecular data.
   - **Temporal split** – data is ordered by `year`; the oldest fraction is train, newest is test. Mimics real-world deployment where models predict on future data.

3. **Cross-Validation (E43)**
   - **5-fold CV** – standard k-fold with shuffling.
   - **Repeated 5-fold CV (×3)** – averages out stochastic split variance by repeating with different random seeds.
   - **Scaffold 5-fold CV** – folds are built from scaffold clusters instead of random partitions.
   - **Nested CV** – outer loop (5 folds) gives an unbiased performance estimate; inner loop (3 folds) selects hyper-parameters (`n_estimators` grid: 50, 100, 200).

4. **Metrics & Uncertainty (E45)**
   - **Point metrics**: R², RMSE, MAE for every fold and every strategy.
   - **Ensemble uncertainty**: variance across fold predictions → 95 % prediction intervals. Coverage and NLL are reported.
   - **Bootstrap uncertainty**: 1,000 bootstrap resamples of the test set to estimate standard error and 95 % CI for each metric.

5. **Comparison & Outputs**
   - Console-printed comparison table (mean ± std per strategy).
   - `synthetic_data.csv` – generated dataset.
   - `evaluation_comparison_summary.csv` – flattened results for quick inspection.
   - `evaluation_results.json` – full structured results (fold-level metrics, uncertainty dicts, runtimes).

## Example Output

```
[2/4] Running simple train/val/test splits ...
       random       | test R² = 0.6123 | RMSE = 1.8912
       scaffold     | test R² = 0.4821 | RMSE = 2.3456
       temporal     | test R² = 0.5102 | RMSE = 2.1987

[3/4] Running cross-validation strategies ...
       5-fold CV        | R² = 0.5987 ± 0.0412 | RMSE = 1.9234 ± 0.0891
       Repeated 5-fold  | R² = 0.6012 ± 0.0389 | RMSE = 1.9156 ± 0.0823
       Scaffold 5-fold  | R² = 0.4654 ± 0.0623 | RMSE = 2.4123 ± 0.1124
       Nested CV        | R² = 0.4521 ± 0.0712 | RMSE = 2.4456 ± 0.1289
```

> **Note**: Exact numbers will vary because the data is synthetic and random.

## Design Notes

- **Scaffold split** is the most important addition for molecular ML because chemically similar molecules often have similar properties; random split can over-estimate performance.
- **Temporal split** is critical when the data distribution drifts over time (e.g. as new PSC device architectures emerge).
- **Nested CV** should be used whenever hyper-parameters are tuned; standard k-fold can be optimistically biased because the same data is used for both model selection and evaluation.
- **Uncertainty quantification** (ensemble variance + bootstrap) helps decide whether a reported performance difference between two strategies is statistically meaningful.

## Extending the Exploration

- Add more models (XGBoost, LightGBM, SVR) by passing different `model_factory` functions.
- Add composition-based splits (E42) for perovskite A/B/X composition clusters.
- Add SHAP / permutation importance (E45 interpretation) to understand which descriptors drive performance differences across split strategies.
- Replace synthetic data with real PSC data by loading `../../data_cache.csv` or the upstream Excel file.

## Data Screening Criteria (数据筛选条件)

> 评估策略必须配合严格的数据筛选才能给出可靠的性能估计。以下为评估前必须应用的筛选条件。

### 评估前数据筛选

| 条件 | 阈值 | 评估影响 |
|------|------|----------|
| 有效SMILES | RDKit可解析 | 无效分子会导致scaffold split失败 |
| Delta_PCE范围 | -5% < ΔPCE < +10% | 极端值会inflate R²（模型学的是异常值） |
| 时间戳完整性 | ≥ 80% 有`year`列 | temporal split需要年份信息 |
| scaffold可计算 | 80%以上分子有唯一scaffold | scaffold split需要足够多样性 |

### 不同Split策略的筛选要求

| Split类型 | 额外筛选条件 | 理由 |
|-----------|-------------|------|
| **Random Split** | 无额外要求 | 最简单，但最乐观 |
| **Scaffold Split** | 删除单scaffold_singleton | 避免一个scaffold只出现一次导致无法分配 |
| **Temporal Split** | 按`year`排序，删除无年份数据 | 确保时间连续性 |
| **Nested CV** | 训练集每fold ≥ 100样本 | 内层CV需要足够数据调参 |

### 评估指标判断标准（文献）

| 指标 | 可接受阈值 | 优秀阈值 | 文献来源 |
|------|-----------|----------|----------|
| CV R² | > 0.30 | > 0.70 | Yang AFM 2025: 0.76 |
| Test RMSE | < 2.0% | < 1.5% | Yang AFM: 1.6% 平均误差 |
| MAE | < 1.5% | < 1.0% | — |
| Bootstrap 95% CI宽度 | < ±0.05 R² | < ±0.02 R² | 统计显著性 |

### 自我判断 checklist

- [ ] Random R² >> Scaffold R²？→ 是，说明模型过拟合到化学相似性
- [ ] Temporal R² < Random R²？→ 是，说明存在时间漂移
- [ ] Nested CV R² ≈ Simple CV R²？→ 否，说明调参导致乐观偏差
- [ ] Bootstrap CI是否包含0？→ 是，说明性能不显著

### 基线PCE对评估策略的影响

> ⚠️ **当前评估未区分基线PCE，导致评估结果失真。**

**问题**：如果训练集和测试集的基线PCE分布不同，模型性能会被错误估计。

| 场景 | 训练集基线PCE | 测试集基线PCE | 评估偏差 |
|------|--------------|--------------|----------|
| 训练集低质量器件多 | mean=8% | mean=18% | 测试R²被人为压低（模型"没想到"器件这么好） |
| 训练集高质量器件多 | mean=18% | mean=8% | 测试R²被人为抬高（模型预测高基线，实际低基线但Delta_PCE大） |

**推荐的评估策略修正**：

1. **分层抽样（Stratified Split）**：按基线PCE分箱后抽样
   ```python
   df["pce_baseline_bucket"] = pd.cut(df["jv_reverse_scan_pce_without_modulator"], bins=[0, 10, 18, 30])
   # 然后按 bucket 分层抽样
   ```

2. **基线PCE作为协变量**：在评估时控制基线PCE的影响
   ```python
   # 计算残差：去掉基线PCE解释的方差后，评估分子特征的贡献
   residuals = y - baseline_pce_effect
   ```

3. **分层评估**：分别报告低/中/高效率器件上的R²
   ```python
   for bucket in ["low", "medium", "high"]:
       r2_bucket = r2_score(y_test[bucket], y_pred[bucket])
   ```

### 当前项目评估结果

| 策略 | 当前项目 | 文献 | 判断 |
|------|----------|------|------|
| Random Split R² | ~0.12 | ~0.76 | ❌ 严重不足 |
| Scaffold Split R² | 未报告 | ~0.65 | ⚠️ 可能更低 |
| Temporal Split R² | 未报告 | ~0.60 | ⚠️ 可能更低 |
| **基线分层评估** | **未实施** | **—** | **❌ 缺失（最关键）** |
| 实验验证 | 无 | 12个验证 | ❌ 缺失 |

**关键发现**：当前评估的根本问题不是CV策略不够复杂，而是**输入特征中缺少基线PCE**。即使使用Nested CV + Scaffold Split，如果模型不知道器件基线质量，预测Delta_PCE仍然是"盲猜"。

**建议的评估流程**：
1. 加入基线PCE作为特征
2. 实施Stratified Split（按基线PCE分层）
3. 同时评估绝对PCE预测（与文献对齐）和Delta_PCE预测（任务目标）
4. 报告分层R²（低/中/高效率器件分别评估）

## References

- AGENTS.md (Layer 4 section) in project root.
- AFM 2025: ML-guided PSC experiments, 2079 data points, R² = 0.76, 12 validation experiments.
- Co-PAS: scaffold pre-screening + JTVAE latent + PubChem 250 k additives.
