# Model Architecture Exploration (Layer 3 — M31)

This folder contains a self-contained exploration of classical ML regressors for the **PSC additive / modulator prediction** task, following the Layer 3 (M31) architecture defined in `AGENTS.md`.

---

## Files

| File | Description |
|------|-------------|
| `explore_models.py` | Main script: loads data, trains each model with 5-fold CV, prints a comparison table, and saves results + plots. |
| `model_factory.py` | Unified model registry / wrapper. Provides a consistent `fit` / `predict` interface across all M31 models. |
| `README.md` | This file. |

---

## Models Evaluated (M31)

| Model | Library | Key Hyper-parameters |
|-------|---------|----------------------|
| Random Forest | `sklearn.ensemble.RandomForestRegressor` | `n_estimators=200`, `max_depth=12` |
| XGBoost | `xgboost.XGBRegressor` | `n_estimators=200`, `max_depth=6`, `lr=0.05` |
| LightGBM | `lightgbm.LGBMRegressor` | `n_estimators=200`, `max_depth=6`, `lr=0.05` |
| SVR | `sklearn.svm.SVR` | `kernel=rbf`, `C=1.0` |
| KNN | `sklearn.neighbors.KNeighborsRegressor` | `n_neighbors=5`, `weights=distance` |

---

## Data Strategy

1. **Primary**: The script first attempts to load `../../data_cache.csv` (the project's cached PSC dataset) and compute **RDKit molecular descriptors** (`MolWt`, `TPSA`, `LogP`, `NumHDonors`, `NumHAcceptors`, `NumRotatableBonds`) from the `smiles` column.
2. **Fallback**: If fewer than 100 valid SMILES are present, the script automatically generates **synthetic PSC-like regression data** (5 354 samples, 10 features) with realistic feature correlations and a non-linear target (`delta_pce` clipped to [-5, +8] %).

---

## Usage

```bash
# Using the system Python that has sklearn, xgboost, lightgbm installed:
/usr/bin/python3.10 explore_models.py

# Or, if your default python has the required packages:
python3 explore_models.py
```

### Requirements
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `pandas`
- `numpy`
- `matplotlib` & `seaborn` (for plots)
- `rdkit` (optional — enables real-data mode)

---

## Outputs (saved in this folder)

| File | Description |
|------|-------------|
| `model_comparison_results.csv` | Aggregated 5-fold CV metrics per model (R², RMSE, MAE means & stds). |
| `per_fold_results.csv` | Fold-by-fold breakdown of every metric. |
| `model_comparison_r2.png` | Bar chart of mean CV R² with error bars. |

---

## Design Notes

- **StandardScaler** is applied per fold (fit on train, transform on validation) because SVR and KNN are distance-based and sensitive to feature scales.
- **Random seed 42** is fixed everywhere for reproducibility.
- The factory pattern in `model_factory.py` makes it trivial to register new models (e.g., CatBoost, ElasticNet) without touching the evaluation logic.

---

## Data Screening Criteria (数据筛选条件)

> 模型训练前的数据筛选条件直接影响模型泛化能力和R²表现。

### 训练集筛选条件

| 条件 | 阈值 | 理由 |
|------|------|------|
| 有效SMILES | RDKit可解析 | 无效分子无法计算描述符 |
| Delta_PCE | -5% < ΔPCE < +10% | 极端异常值会破坏模型学习 |
| PCE_without_modulator | 5% < PCE_base < 25% | 基础器件性能必须在合理范围内 |
| 分子量 | 100–800 Da | 物理可实现的添加剂范围 |
| 训练集大小 | ≥ 500 samples | 保证模型有足够数据学习 |

### 特征缩放与标准化

| 条件 | 方法 | 理由 |
|------|------|------|
| 距离敏感模型 | `StandardScaler` (per fold) | SVR/KNN对特征尺度敏感 |
| 树模型 | 无需缩放 | RF/XGBoost对尺度不敏感 |
| 异常值处理 | `RobustScaler` (可选) | 对极端值更稳健的缩放 |

### 模型选择标准（文献验证）

| 模型 | 文献R² | 适用场景 | 当前状态 |
|------|--------|----------|----------|
| **Random Forest** | 0.76 (Yang AFM) | 实验数据，小-中样本 | ✅ 已实现 |
| **XGBoost** | 0.9999 (SCAPS模拟) | 模拟数据，大样本 | ✅ 已实现 |
| **CatBoost** | ≥0.88 (Obada) | bandgap预测，大样本 | ❌ 未实现（推荐添加） |
| **LightGBM** | ~0.70+ | 快速训练，大样本 | ✅ 已实现 |
| **GPR** | ~0.95 (ACS Omega) | 小样本，不确定性量化 | ❌ 未实现（关键缺失） |
| **SVR** | — | 小样本，非线性 | ✅ 已实现 |

### 基线PCE作为输入特征（最关键改进）

> ⚠️ **当前模型的最大缺陷：输入特征中没有基线PCE (`jv_reverse_scan_pce_without_modulator`)，却要求预测Delta_PCE。**

这相当于让模型"盲猜"器件质量。同一分子对低质量器件（PCE=5%）和高质量器件（PCE=20%）的Delta_PCE完全不同。

**数据证据**：
- 基线PCE与Delta_PCE相关系数 r = -0.33
- 基线PCE均值=11.83%，范围=0–30%
- 不控制基线PCE，模型学到的主要是"器件质量偏见"而非分子特征

**立即修改建议**：

```python
# 在 model_factory.py 或 explore_models.py 中
FEATURE_COLS = [
    "molecular_weight", "h_bond_donors", "h_bond_acceptors",
    "rotatable_bonds", "tpsa", "log_p",
    "jv_reverse_scan_pce_without_modulator",  # ← 必须加入
]
```

**同时建议尝试预测绝对PCE**（与文献对齐）：

```python
# 方案A：预测绝对PCE（文献主流）
target = "jv_reverse_scan_pce"
features = [..., "jv_reverse_scan_pce_without_modulator"]
# 预期R²：0.50–0.70

# 方案B：保留Delta_PCE，但加入基线PCE作为特征
target = "delta_pce"
features = [..., "jv_reverse_scan_pce_without_modulator"]
# 预期R²：0.30–0.50（从当前0.16提升）
```

### 关键差距与改进

- **当前最佳CV R²**: 0.16 (WeightedEnsemble_L2, 预测Delta_PCE)
- **文献金标准**: 0.76 (Yang AFM, **预测绝对PCE**, 2079实验器件)
- **核心问题**: 
  1. 目标变量选择：Delta_PCE比绝对PCE更难预测
  2. 特征缺失：基线PCE未作为输入特征
  3. 数据信噪比低：差分变量放大测量误差
- **优先级**: 
  1. **立即加入基线PCE作为特征**（最快提升）
  2. 尝试预测绝对PCE（与文献对齐）
  3. 添加CatBoost / GPR
  4. 添加器件/工艺特征
