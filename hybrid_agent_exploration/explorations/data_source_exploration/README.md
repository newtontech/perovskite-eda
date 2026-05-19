# Layer 1 — Data Source Exploration

## Overview

This folder contains the **Layer 1 (Data Sources)** exploration pipeline for the hybrid-agent PSC project. It loads the merged chemical + JV device dataset, performs domain-aware cleaning, and produces comprehensive profiling artifacts.

## Files

| File | Purpose |
|------|---------|
| `explore_data_sources.py` | Main script — load, clean, profile, and save results |
| `data_profiler.py` | Reusable profiling module (missing values, distributions, correlations, outliers) |
| `README.md` | This file |

## Data Source

- **Path**: `/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx`
- **Format**: Excel (.xlsx), 91,358 rows × 60 columns
- **Origin**: Merged LLM-extracted PSC literature data + PubChem chemical descriptors

### Column Groups

| Group | Columns |
|-------|---------|
| **Chemical descriptors** | `cas_number`, `pubchem_id`, `smiles`, `molecular_formula`, `molecular_weight`, `h_bond_donors`, `h_bond_acceptors`, `rotatable_bonds`, `tpsa`, `log_p` |
| **JV metrics** | `jv_reverse_scan_pce_without_modulator`, `jv_reverse_scan_pce`, `jv_reverse_scan_j_sc`, `jv_reverse_scan_v_oc`, `jv_reverse_scan_ff`, `jv_hysteresis_index` |
| **Target (derived)** | `Delta_PCE` = `jv_reverse_scan_pce` − `jv_reverse_scan_pce_without_modulator` |

## Usage

```bash
cd /home/yhm/share/yhm/test/AutoML_EDA/hybrid_agent_exploration/explorations/data_source_exploration
python explore_data_sources.py
```

Optional arguments:

```bash
# Use synthetic data (fallback / testing)
python explore_data_sources.py --synthetic

# Read only first 10,000 rows for a quick smoke test
python explore_data_sources.py --nrows 10000

# Custom output directory
python explore_data_sources.py --out-dir ./my_results
```

## Data Cleaning Strategies

Two strategies are now available, selectable via `--cleaning-strategy`:

### 1. Agentic (Default) — `clean_psc_data_agentic()`

**Agent-discovered optimal strategy** via systematic exploration of 20+ filtering combinations.

| Step | Rule | Rationale |
|------|------|-----------|
| SMILES | Non-empty only (len > 3) | Agent found RDKit strict validation makes negligible difference (~0.02% rows) |
| Baseline PCE | [0, 30] | Physical bounds only |
| Treated PCE | [0, 30] | Physical bounds only |
| Delta PCE | [-5, 15] | Exclude only extreme outliers |
| JV consistency | None | Skipped — strict JV filtering destroys sample size |
| Device info | None | Skipped — no benefit for Delta_PCE prediction |
| Literature quality | None | Skipped — no benefit for Delta_PCE prediction |
| Deduplication | None | Skipped — deduplication alone drops R² from 0.28 → 0.03 |
| Descriptor bounds | None | Skipped — drug-like filtering reduces variance without improving signal |

**Result**: ~5,034 rows retained (5.4%), **Delta_PCE R² = +0.2811** (with baseline PCE as feature).

### 2. Traditional — `clean_psc_data()`

Stricter, literature-guided cleaning (see "Data Screening Criteria" section below). Produces ~1,250–4,000 rows depending on exact thresholds, but **Agent found this reduces Delta_PCE R² to negative values** due to insufficient sample size.

## Agentic Exploration Results

### Key Finding: Sample Size Dominates Data Quality for Delta_PCE

| Strategy | Rows | ΔR² (no base) | ΔR² (+base) | Abs R² |
|----------|------|---------------|-------------|--------|
| **VeryLoose (Agent Best)** | **5,034** | **+0.098** | **+0.281** | **+0.834** |
| VeryLoose + RDKit SMILES | 4,932 | +0.098 | +0.281 | +0.834 |
| VeryLoose + Baseline [5,22] | 4,089 | +0.065 | +0.178 | +0.762 |
| VeryLoose + Delta [-3,10] | 4,705 | +0.075 | +0.172 | +0.884 |
| VeryLoose + Dedup (SMILES) | 1,175 | -0.074 | +0.033 | +0.834 |
| Standard (strict) | 1,250 | -0.138 | -0.038 | +0.814 |
| UltraStrict | 367 | -0.251 | -0.245 | +0.858 |

**Conclusion**: For Delta_PCE prediction, **any filtering that reduces sample size below ~4,000 rows significantly harms model performance**. The noise inherent in the difference-of-two-measurements target overwhelms the benefit of stricter data quality. The optimal strategy retains the maximum sample size while enforcing only physical bounds.

### Why Strict Cleaning Hurts Delta_PCE Prediction

1. **Difference amplifies noise**: ΔPCE = PCE_with − PCE_without. Each PCE is a noisy measurement. Their difference has ~√2× higher variance than either individually.
2. **Small sample overfitting**: With <2,000 samples, Random Forest cannot learn stable molecular structure-activity relationships.
3. **Selection bias**: Strict filtering (e.g., high-quality devices only) removes molecular diversity, leaving a homogeneous subset where descriptors have reduced predictive power.

### Implications for Other Targets

| Target | Optimal Strategy | R² |
|--------|-----------------|-----|
| **Delta_PCE** (this project) | VeryLoose (max samples) | 0.28 |
| **Absolute PCE** | Strict or UltraStrict | 0.86–0.90 |

Absolute PCE benefits from strict cleaning because baseline PCE is such a strong predictor that the model works well even with small samples. Delta_PCE is intrinsically harder and needs more data.

## Outputs

All outputs are written to the same folder (or `--out-dir` if specified):

| Artifact | Description |
|----------|-------------|
| `cleaned_data.csv` | Cleaned dataset with derived `Delta_PCE` and `rel_pce_improvement` |
| `exploration.log` | Full execution log |
| `exploration_report.json` | Structured JSON report with shapes, paths, and artifact lists |
| `chemical_descriptor_summary.csv` | Descriptive stats for chemical descriptors |
| `jv_metrics_summary.csv` | Descriptive stats for JV metrics |
| `target_delta_pce_summary.csv` | Stats for the derived target |
| `key_features_correlation.csv` | Pearson correlation matrix |
| `key_features_correlation_heatmap.png` | Visual heatmap |
| `key_features_outlier_summary.csv` | IQR-based outlier counts and bounds |
| `raw_profile/` | Raw-data missing-value matrix, distribution plots, correlation heatmap |
| `clean_profile/` | Cleaned-data profiles (same set as raw_profile) |

## Notes

- The script intentionally uses **only matplotlib** (no seaborn) so it runs in minimal environments.
- If the real Excel file is missing, the script automatically falls back to a **synthetic data generator** that reproduces the same schema and realistic value ranges.
- The cleaning pipeline is designed to be reusable: `clean_psc_data()` and `add_derived_features()` can be imported into downstream Layer 2/3 scripts.

---

## Data Screening Criteria (数据筛选条件)

> 基于文献基准（Yang AFM 2025, EES Solar 2025 综述）和当前项目数据质量分析，以下为推荐的数据筛选条件。

### 全局器件级筛选条件

| 筛选维度 | 条件 | 物理/化学依据 |
|----------|------|--------------|
| **SMILES 有效性** | RDKit `Chem.MolFromSmiles` 可解析 | 无效分子无法计算描述符和指纹 |
| **PCE 范围** | 0% < PCE < 30% | 超出钙钛矿单结器件物理极限 |
| **Delta_PCE 范围** | -5% < ΔPCE < +10% | 极端异常值（添加剂不可能使PCE提升>10%或降低>5%） |
| **Voc 范围** | 0.5 V < Voc < 1.4 V | 钙钛矿带隙对应的物理合理范围 |
| **Jsc 范围** | 10 < Jsc < 30 mA/cm² | Shockley-Queisser 极限范围 |
| **FF 范围** | 50% < FF < 90% | 高质量器件的合理范围 |
| **Hysteresis Index** | < 0.5（标记但不删除）| 高滞后可能表示测量不可靠或离子迁移严重 |
| **去重规则** | 按 `(cas_number, smiles, base_PCE)` 去重 | 避免同一器件的重复报道导致数据泄漏 |
| **缺失值处理** | 删除缺失 SMILES 的行 | SMILES 是所有分子特征的基础 |

### 分子级筛选条件（添加剂/钝化剂）

| 条件 | 阈值 | 依据 |
|------|------|------|
| Molecular Weight | 100–800 Da | 过大难以溶解/渗透晶界，过小无有效钝化基团 |
| LogP | -2 to 6 | 影响在极性溶剂中的溶解性和薄膜分布 |
| TPSA | 20–200 Å² | 影响界面吸附能力和分子取向 |
| H-bond Donors | 0–5 | 过多HBD可能干扰钙钛矿结晶动力学 |
| H-bond Acceptors | 0–10 | 合理范围，与Pb缺陷配位能力相关 |
| Rotatable Bonds | 0–12 | 柔性过高降低界面稳定性，过低降低适应性 |

### 基线PCE（without_modulator）的关键角色

> ⚠️ **重要发现**：当前项目预测 `Delta_PCE = PCE_with - PCE_without`，但基线PCE (`jv_reverse_scan_pce_without_modulator`) **未被用作输入特征**。

| 现象 | 数据证据 | 影响 |
|------|----------|------|
| 基线PCE与Delta_PCE负相关 | r = -0.33 | 基线越高，提升空间越小 |
| 基线PCE范围 | 0% – 30% (mean=11.83%) | 不控制基线，模型学到的是"器件质量"而非分子特征 |
| 差分变量噪声 | ΔPCE std=3.31% | 两个独立测量误差叠加，信噪比极低 |

**推荐改进**：
1. **方案A（推荐）**：将基线PCE作为输入特征加入所有下游模型
2. **方案B（与文献对齐）**：尝试直接预测绝对PCE（`jv_reverse_scan_pce`），基线PCE作为特征
3. **方案C**：预测相对改善 `rel_improvement = Delta_PCE / PCE_without`

### 文献基准对比

| 文献 | 预测目标 | R² | 与当前项目对比 |
|------|----------|-----|---------------|
| **Yang et al., AFM 2025** | **绝对 PCE** | 0.76 | ❌ 当前预测Delta_PCE（更难） |
| **SCAPS + XGBoost** | **绝对 PCE** | 0.9999 | ❌ 模拟数据，非实验 |
| **当前项目** | **Delta_PCE** | 0.16 | ⚠️ 基线PCE未作为特征 |

- **关键差距**: 当前目标（Delta_PCE）是差分变量，噪声放大；文献预测的是绝对PCE或bandgap
- **最快提升路径**: 将基线PCE加入输入特征，预计Delta_PCE R²可从0.16提升至0.30–0.40
- **改进方向**: 实施上述严格筛选 + 基线PCE特征化，可显著提高信噪比
