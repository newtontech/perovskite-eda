# Layer 2 Feature Engineering Exploration

> Path: `explorations/feature_engineering_exploration/`
>
> Focus: Molecular representations for perovskite additive / modulator screening (F21 & F22)

## Overview

This folder contains a self-contained exploration of **Layer 2 — Representations** from the hybrid-agent pipeline. It systematically generates and compares multiple molecular feature sets, analyses their dimensionality, sparsity, variance, and predictive signal via univariate selection.

## Files

| File | Purpose |
|------|---------|
| `explore_features.py` | Main runnable script. Loads data, generates descriptors & fingerprints, runs analyses, saves artefacts. |
| `feature_generators.py` | Modular feature-generation library (F21 descriptors, F22 fingerprints). Can be imported independently. |
| `README.md` | This file. |

## Feature Sets Generated

### F21 — Molecular Descriptors (RDKit)

| Sub-set | # Features | Description |
|---------|-----------|-------------|
| `Descriptors_basic` | 12 | Hand-picked: MolWt, LogP, TPSA, HBD, HBA, RotBonds, AromaticRings, HeavyAtoms, RingCount, FractionCSP3, NumValenceElectrons, NumHeteroatoms |
| `Descriptors_full` | ~200 | Complete RDKit `Descriptors.descList` (all 2-D molecular descriptors) |

### F22 — Molecular Fingerprints

| Name | Bits | Type |
|------|------|------|
| `FP_ECFP4` | 2,048 | Morgan circular fingerprint, radius=2 (ECFP4) |
| `FP_ECFP6` | 2,048 | Morgan circular fingerprint, radius=3 (ECFP6) |
| `FP_MACCS` | 167 | MACCS structural keys |
| `FP_KRFP` | 2,048 | Hashed atom-pair fingerprint |
| `FP_AtomPair` | 2,048 | Hashed atom-pair fingerprint (explicit API alias) |
| `FP_TopologicalTorsion` | 2,048 | Hashed topological-torsion fingerprint |

## Quick Start

```bash
cd explorations/feature_engineering_exploration
python explore_features.py
```

### What it does

1. **Load data** — reads `../../data_cache.csv` (SMILES + target `delta_pce`). Falls back to a synthetic dataset of 500 molecules if the cache is missing.
2. **Filter valid molecules** — keeps only SMILES that RDKit can parse.
3. **Generate features** — computes all descriptor and fingerprint sets above.
4. **Compare dimensionality** — records `n_samples × n_features`, NaN fraction, and zero-bit density for each set.
5. **Variance analysis** — runs `VarianceThreshold` at 0.00, 0.01, 0.05 to show how many features survive.
6. **Univariate selection** — ranks top-20 features per set via `f_regression` against `delta_pce`.
7. **Visualise** — saves:
   - `feature_set_dimensionality.png` — bar chart of feature counts
   - `correlation_heatmap_Descriptors_*.png` — top-30 descriptor correlation matrices
   - `fingerprint_densities.png` — bit-density histograms for fingerprint sets
8. **Persist artefacts** — writes:
   - `feature_exploration_report.json` — structured summary
   - `*.csv.gz` — compressed feature matrices
   - `top_features_*.csv` — ranked feature lists

## Outputs

All outputs are written to the same folder (`explorations/feature_engineering_exploration/`).

```
explorations/feature_engineering_exploration/
├── explore_features.py
├── feature_generators.py
├── README.md
├── feature_exploration_report.json
├── feature_set_dimensionality.png
├── correlation_heatmap_Descriptors_basic.png
├── correlation_heatmap_Descriptors_full.png
├── fingerprint_densities.png
├── Descriptors_basic.csv.gz
├── Descriptors_full.csv.gz
├── FP_ECFP4.csv.gz
├── FP_ECFP6.csv.gz
├── FP_MACCS.csv.gz
├── FP_KRFP.csv.gz
├── FP_AtomPair.csv.gz
├── FP_TopologicalTorsion.csv.gz
├── top_features_Descriptors_basic.csv
├── top_features_Descriptors_full.csv
├── top_features_FP_ECFP4.csv
└── ... (etc.)
```

## Design Notes

- **Reusability**: `feature_generators.py` is import-safe (adds project root to `sys.path`) and exposes a clean `generate_all_features(smiles_series)` API.
- **Graceful degradation**: Invalid SMILES are handled silently — descriptors become `NaN` and fingerprints become all-zero vectors.
- **Scalability**: Fingerprints are stored as `int8` dense arrays. For >10k molecules the `.csv.gz` files remain manageable (~1–5 MB per 2k-bit set).
- **Determinism**: Fixed `RANDOM_STATE` for synthetic data; RDKit fingerprints are deterministic by design.

## Dependencies

- `rdkit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

All are already listed in the project `requirements.txt`.

## Data Screening Criteria (数据筛选条件)

> 特征工程层的数据筛选直接影响下游模型性能。以下为针对分子特征的筛选条件。

### 输入数据筛选（进入特征工程前）

| 条件 | 阈值 | 理由 |
|------|------|------|
| SMILES 有效性 | RDKit 可解析 | 无效SMILES无法生成任何分子特征 |
| 分子量 | 100–800 Da | 超出范围的分子可能不是有效的添加剂 |
| LogP | -2 to 6 | 极端LogP值可能导致薄膜形貌失控 |
| 重金属元素 | 排除含 Pb, Cd, Hg 的有机分子 | 有机添加剂不应含重金属 |

### 特征级筛选（特征生成后）

| 条件 | 方法 | 理由 |
|------|------|------|
| 低方差特征 | `VarianceThreshold(threshold=0.01)` | 移除近常数特征，降低过拟合风险 |
| 高相关性特征 | Pearson correlation > 0.95 | 移除冗余特征，减少多重共线性 |
| 缺失率 > 30% | 删除该特征列 | 缺失过多无法可靠填充 |
| 单变量选择 | `SelectKBest(f_regression, k=20)` | 保留与Delta_PCE最相关的特征 |

### 指纹特殊处理

| 条件 | 阈值 | 理由 |
|------|------|------|
| 全零指纹比例 | < 90% | 全零率过高说明SMILES无效或过于简单 |
| 位密度（bit density） | 1%–20% | ECFP理想密度；过高=无区分度，过低=信息不足 |

### 基线PCE特征工程（关键缺失）

> ⚠️ **当前所有特征都是分子级描述符，缺少器件级特征，尤其是基线PCE。**

| 特征类型 | 当前状态 | 重要性 | 建议 |
|----------|----------|--------|------|
| 分子描述符（MolWt, LogP, TPSA...） | ✅ 已有 | 中等 | 保留 |
| **基线PCE** (`jv_reverse_scan_pce_without_modulator`) | ❌ 缺失 | **极高** | **必须加入** |
| 器件结构（n-i-p / p-i-n） | ❌ 缺失 | 高 | 编码为分类特征 |
| 钙钛矿组分（A/B/X ratio） | ❌ 缺失 | 高 | 提取 tolerance factor, octahedral factor |
| 工艺参数（退火温度、浓度、溶剂） | ❌ 缺失 | 高 | 从文本中解析 |
| DFT描述符（HOMO/LUMO, dipole） | ❌ 缺失 | 高 | 通过RDKit或DFT计算 |

**基线PCE的特征处理建议**：

```python
# 1. 原始值作为连续特征
features["pce_baseline"] = df["jv_reverse_scan_pce_without_modulator"]

# 2. 分箱编码（捕捉非线性）
def pce_bucket(x):
    if x < 10: return "low"
    elif x < 18: return "medium"
    else: return "high"

# 3. 交互特征（基线PCE × 分子特征）
features["pce_baseline_x_molwt"] = df["pce_baseline"] * df["molecular_weight"]
features["pce_baseline_x_logp"] = df["pce_baseline"] * df["log_p"]
```

### 文献对标

- **EES Solar 2025 综述**: 特征选择和特征提取是关键预处理步骤；PCA/PLS常用
- **Co-PAS**: scaffold预筛选 + JTVAE latent + PubChem 25万分子做添加剂筛选
- **当前问题**: 仅用12个basic descriptors；缺少DFT描述符（HOMO/LUMO, dipole, ESP）
- **关键发现**: **文献中预测的是绝对PCE（R²=0.76），不是Delta_PCE**。当前预测Delta_PCE更难，且未利用基线PCE信息
- **改进方向**: 
  1. 立即加入基线PCE作为特征（预期R²提升0.15–0.25）
  2. 添加器件结构特征 + 钙钛矿组分特征 + DFT量子特征

## References

- F21 / F22 definitions: see `../../AGENTS.md` → Layer 2 Representations
- Existing implementations: `../../features/rdkit_descriptors.py`, `../../features/fingerprints.py`
