# 文献基准对比与当前项目差距分析

> 基于 AI/ML for Perovskite Solar Cells (PSC) 最新文献（2024–2026）
> 生成时间：2026-04-30

---

## 1. 顶级文献基准汇总

### 1.1 实验数据直接预测（金标准）

| 文献 | 年份 | 任务 | 数据量 | 最佳模型 | R² | 验证方式 | 实验验证 |
|------|------|------|--------|----------|-----|----------|----------|
| **Yang et al., AFM** | 2025 | PCE预测 | 2,079 实验器件 | RF/GB | **0.76** | 5-fold CV + 外部测试集 | **12个验证实验**，平均误差 1.6% |
| **Liang et al.** | 2025 | PCE预测（宽禁带） | 未公开 | Gradient Boosting + SHAP | ~0.70+ | CV + 实验验证 | 实验验证 21.81% PCE |
| **GPR + SCAPS (ACS Omega)** | 2023 | 器件参数优化 | ~50 模拟点 | Gaussian Process | ~0.95 | LOO CV | 实验验证，误差 0.11–1.47% |

### 1.2 模拟/计算数据预测（高 R² 但非实验）

| 文献 | 年份 | 任务 | 数据量 | 最佳模型 | R² | 验证方式 |
|------|------|------|--------|----------|-----|----------|
| **SCAPS-1D + XGBoost** | 2025 | 多参数器件模拟 | 数千模拟点 | XGBoost | **0.9999** | Train/Test Split |
| **Vakharia et al.** | 2022 | Bandgap预测 | 129 钙钛矿 | LR (ElasticNet) | **0.98** | CV |
| **Obada et al.** | 2024 | Bandgap预测 | 大型数据集 | CatBoost | **≥0.88** | CV |
| **Taeseo et al. (迁移学习)** | 2025 | Bandgap预测 | DFT数据 | 迁移学习 | **0.817** | CV |

### 1.3 关键洞察（来自 EES Solar 2025 综述）

- **模型选择**：RF 和 Gradient Boosting（LGBM, XGBoost, CatBoost）在 PSC 预测中表现最佳
- **特征工程**：特征选择和特征提取是关键预处理步骤；PCA/PLS 常用
- **评估指标**：必须同时使用 R² 和 RMSE；高 R² 但大 RMSE 仍无科学意义
- **不确定性**：GPR 提供不确定性估计，对实验闭环至关重要
- **外部验证**：实验验证是 PSC 领域的金标准
- **数据划分**：scaffold split / temporal split 比 random split 更严格

---

## 2. 当前项目结果 vs 文献基准

### 2.1 当前项目状态（AutoGluon, 5-fold CV, basic chemical descriptors）

| 指标 | 当前值 | 文献基准 | 差距 |
|------|--------|----------|------|
| **CV R²** | 0.1624 | 0.76 (Yang AFM) | **↓ 4.7×** |
| **Test R²** | 0.1201 | 0.76 | **↓ 6.3×** |
| **Test RMSE** | 3.402% | 1.6% (Yang AFM) | **↑ 2.1×** |
| **实验验证** | 无 | 12个验证实验 | **缺失** |
| **不确定性量化** | 无 | GPR/Ensemble | **缺失** |
| **数据量** | 5,354 | 2,079 (Yang) | 相当 |

### 2.2 核心差距分析

#### 差距 1：目标变量难度
- **文献**：预测绝对 PCE 或 Bandgap —— 这些是器件/材料的本征属性
- **本项目**：预测 **Delta_PCE = PCE_with_modulator - PCE_without_modulator**
  - 这是**差分变量**，两个测量值的误差叠加
  - 包含大量**负数据**（添加剂反而降低性能）
  - 信噪比极低，是文献中最难的任务类型

#### 差距 2：特征不足
- **文献使用的特征**：
  - 器件结构（n-i-p / p-i-n）
  - 传输层材料
  - 钙钛矿组分（A/B/X ratio, tolerance factor, octahedral factor）
  - 工艺参数（溶剂、退火温度、浓度、旋涂转速）
  - DFT描述符（HOMO/LUMO, dipole, ESP）
- **本项目当前特征**：
  - 仅 basic chemical descriptors（12个）
  - 缺少器件结构、工艺参数、组分特征

#### 差距 3：数据质量问题
- 原始 91,357 → 5,354（完整率仅 5.86%）
- 大量缺失 SMILES、缺失 JV 数据
- Delta_PCE 分布可能包含极端异常值
- 没有进行 scaffold split / temporal split 验证

#### 差距 4：缺少严格的验证闭环
- **文献**：CV → 外部测试 → 实验验证 → 误差分析
- **本项目**：仅有 CV，无外部实验验证
- 无不确定性估计（GPR / ensemble variance）

---

## 3. 五套管线是否足以达到文献标准？

### 3.1 能力对照表

| 文献要求 | Layer 1 (数据) | Layer 2 (特征) | Layer 3 (模型) | Layer 4 (评估) | Layer 5 (部署) | 当前状态 |
|----------|---------------|---------------|---------------|---------------|---------------|----------|
| 高质量数据清洗 | ✅ | — | — | — | — | 部分实现 |
| 器件/工艺特征 | — | ❌ | — | — | — | **缺失** |
| DFT/量子特征 | — | ❌ | — | — | — | **缺失** |
| RF/XGBoost/CatBoost | — | — | ✅ | — | — | 已实现 |
| GPR + 不确定性 | — | — | ❌ | — | — | **缺失** |
| Scaffold/Temporal Split | — | — | — | ✅ | — | 已实现 |
| Nested CV + BO | — | — | — | ✅ | — | 部分实现 |
| 外部实验验证 | — | — | — | — | ❌ | **缺失** |

### 3.2 关键结论

> **当前5套管线组合尚不足以迅速达到文献 R² = 0.76 的标准。**

原因：
1. **特征层（Layer 2）严重缺失**：只有分子描述符，缺少器件结构、工艺参数、钙钛矿组分特征、DFT描述符
2. **模型层（Layer 3）缺少 GPR**：无法提供不确定性估计，无法实现主动学习闭环
3. **部署层（Layer 5）无实验验证**：无法像 Yang et al. 那样做12个验证实验
4. **数据层（Layer 1）筛选不足**：需要更严格的筛选条件来提高信噪比

---

## 4. 数据筛选条件建议（各层应记录）

### 4.1 全局数据筛选条件

| 筛选维度 | 条件 | 理由 |
|----------|------|------|
| **SMILES 有效性** | RDKit 可解析 | 无效分子无法计算描述符 |
| **PCE 范围** | 0% < PCE < 30% | 超出物理合理范围的异常值 |
| **Delta_PCE 范围** | -5% < Delta_PCE < +10% | 极端异常值 |
| **Voc 范围** | 0.5V < Voc < 1.4V | 物理合理性 |
| **Jsc 范围** | 10 < Jsc < 30 mA/cm² | 物理合理性 |
| **FF 范围** | 50% < FF < 90% | 物理合理性 |
| **Hysteresis Index** | < 0.5（标记）| 高滞后可能表示测量不可靠 |
| **Duplicate Removal** | 按 (cas_number, smiles, base_PCE) 去重 | 避免数据泄漏 |
| **Scaffold Split** | 训练/测试集结构分离 | 严格评估泛化性 |
| **Temporal Split** | 按发表时间划分 | 模拟真实部署场景 |

### 4.2 分子级筛选条件（添加剂/钝化剂）

| 条件 | 阈值 | 理由 |
|------|------|------|
| Molecular Weight | 100–800 Da | 过大难以溶解，过小无钝化效果 |
| LogP | -2 to 6 | 影响薄膜形貌 |
| TPSA | 20–200 Å² | 影响界面吸附 |
| H-bond Donors | 0–5 | 过多可能影响结晶 |
| H-bond Acceptors | 0–10 | 合理范围 |
| Rotatable Bonds | 0–12 | 柔性影响组装 |

---

## 5. 改进路线图（达到文献标准）

### 阶段 1：数据质量提升（目标 R² ≈ 0.30–0.40）
- [ ] 实施严格的数据筛选（见4.1）
- [ ] 添加器件结构特征（n-i-p/p-i-n, ETL/HTL材料）
- [ ] 添加工艺参数特征（浓度、退火温度、溶剂）

### 阶段 2：特征工程增强（目标 R² ≈ 0.50–0.60）
- [ ] 添加钙钛矿组分特征（A/B/X ratio, tolerance factor）
- [ ] 使用 ECFP + MACCS 组合指纹
- [ ] 特征选择（Variance Threshold, SelectKBest, SHAP）

### 阶段 3：模型升级（目标 R² ≈ 0.65–0.75）
- [ ] 引入 CatBoost / LightGBM / XGBoost 调优
- [ ] 添加 GPR 作为不确定性基准
- [ ] AutoGluon 自动集成（已有，但需更好特征）

### 阶段 4：严格验证与闭环（目标 R² ≥ 0.76）
- [ ] Scaffold Split + Temporal Split 验证
- [ ] Nested CV + Bayesian Optimization
- [ ] 外部实验验证（如条件允许）
- [ ] 不确定性量化 + 主动学习

---

## 6. 参考文献

1. Yang A. et al., "Enhancing Power Conversion Efficiency of Perovskite Solar Cells Through Machine Learning Guided Experimental Strategies", **Advanced Functional Materials**, 2025. (R²=0.76, 12验证实验)
2. EES Solar Review 2025, "Machine learning for perovskite solar cells", **RSC EES Solar**, 2025.
3. Vakharia et al., LR bandgap prediction, **Elsevier**, 2022. (R²=0.98)
4. Obada et al., CatBoost bandgap prediction, 2024. (R²≥0.88)
5. Taeseo et al., Transfer learning bandgap prediction, 2025. (R²=0.817)
6. SCAPS-1D + XGBoost, **F1000Research**, 2025. (R²=0.9999, 模拟数据)
7. Liang R. et al., "Accelerating Perovskite Solar Cell Development: A Machine Learning-Driven Framework with SHAP Explainability", 2025.
8. GPR + diode physics, **ACS Omega**, 2023.
