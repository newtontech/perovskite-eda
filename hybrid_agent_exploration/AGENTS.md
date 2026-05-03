# Hybrid Agent Exploration — AGENTS.md

> 钙钛矿太阳能电池（PSC）分子添加剂/组分/工艺优化的 AI Agent 自主探索方案
> 路径: `/share/yhm/test/AutoML_EDA/hybrid_agent_exploration/`
> 聚焦: PSC 分子添加剂、钝化剂、HTM/ETM/SAM、A/B/X组分、溶剂与工艺配方

---

## 项目定位

当前钙钛矿 ML 文献的共同问题：每篇论文只选择一条路径（一个数据源 + 一种特征 + 一个模型 + 一种评估），然后一路走到头。

**核心观点**：AI Agent 的价值不是替代某个模型，而是自动探索和调试跨层组合路径，并把数据库、DFT、代码、实验和新数据回流连接成闭环。

**文献趋势正在从**："表格数据 + 手工特征 + RF/XGBoost/SVR"
**扩展到**："JTVAE/Uni-Mol 等学习表示 + Bayesian/主动学习 + 实验验证/自驱闭环"

---

## 五层管线架构

### Layer 1 — 数据来源 (Data Sources)

| 编号 | 数据源 | 规模/格式 | 说明 |
|------|--------|-----------|------|
| S11 | 文献与人工整理数据 | PSC device tables, JV/EQE/stability, additive/passivator reports, negative data | 本项目 D5 属此类 |
| S12 | 钙钛矿公共数据库 | Perovskite Database Project (42,400+器件), PerovskiteDB, Materials Project, NOMAD/OQMD/JARVIS | FAIR原则 |
| S13 | 文本挖掘数据 | NLP from papers, fabrication details, device stack extraction | 本项目 Perovskite_Database_Multiagents 的核心能力 |
| S14 | 计算数据 | DFT/MD, high-throughput calculation, SCAPS/drift-diffusion | 量子化学描述符来源 |
| S15 | 化学分子库 | PubChem (1.1亿), ChEMBL, ZINC, DrugBank-like molecules | 虚拟筛选的分子池 |
| S16 | 可购买分子库 | Sigma-Aldrich, TCI, Aladdin, Energy Chemical, Enamine, eMolecules, Mcule | 确保候选分子可购买 |
| S17 | 自动化实验数据 | robotic synthesis, blade coating/slot-die, in situ PL/imaging, self-driving lab logs | 闭环数据回流 |

**本项目的数据源**: `S11/S13` — `/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx`

数据列：
- **化学数据**: cas_number, pubchem_id, smiles, molecular_formula, molecular_weight, h_bond_donors, h_bond_acceptors, rotatable_bonds, tpsa, log_p
- **JV数据**: jv_reverse_scan_pce_without_modulator, jv_reverse_scan_pce, jv_reverse_scan_j_sc, jv_reverse_scan_v_oc, jv_reverse_scan_ff, jv_hysteresis_index 等
- **目标变量**: Delta_PCE = jv_reverse_scan_pce - jv_reverse_scan_pce_without_modulator
- **样本量**: 原始 91,357 → 完整案例 5,354 (5.86%)

---

### Layer 2 — 数据分析与表征 (Representations)

| 编号 | 方法 | 维度 | 说明 |
|------|------|------|------|
| F21 | 分子描述符 (RDKit/Mordred/PaDEL) | 200-1800+ | donor number, HBA/HBD, LogP 等 |
| F22 | 分子指纹 (Morgan/ECFP/MACCS/KRFP) | 166-4860-bit | 亚结构指纹 |
| F23 | DFT/量化特征 | 变长 | HOMO/LUMO, band gap, dipole, ESP, charge, adsorption/formation energy |
| F24 | 钙钛矿组分特征 | 变长 | A/B/X composition, ionic radius, tolerance factor, octahedral factor, electronegativity |
| F25 | 器件与工艺特征 | 变长 | architecture (n-i-p/p-i-n), transport layers, solvent, annealing, additive concentration, initial PCE |
| F26 | 学习型表示 | 128-512 dim | SMILES token, molecular graph embedding, crystal graph, **JTVAE latent**, **Uni-Mol 3D embedding** |
| F27 | 图像/谱学表征 | 变长 | PL/TRPL/UV-vis, XRD/GIWAXS, SEM/AFM, in situ PL/reflection |

**本项目已实现**: F21 (RDKit Descriptors), F22 (ECFP4, MACCS, KRFP, AP, TT), F25 (部分工艺特征)

**前沿趋势**: F26 学习型表示（JTVAE latent, Uni-Mol embedding）正在取代手工特征。Co-PAS 已用 scaffold 预筛选 + JTVAE latent + PubChem 25万分子做添加剂筛选。

---

### Layer 3 — 机器学习模型 (ML Models)

| 编号 | 模型类别 | 代表方法 | 适用场景 |
|------|----------|----------|----------|
| M31 | 经典ML | Random Forest, XGBoost/LightGBM, SVR/KNN, Elastic Net/Ridge/Lasso | 小数据集基线，多篇PSC文献验证 |
| M32 | 深度学习 | MLP/DNN, CNN for images/spectra, GNN/MPNN, CGCNN/MEGNet | 大数据集，图像/谱学/晶体图 |
| M33 | 预训练分子模型 | Uni-Mol, MolCLR, ChemBERTa, MolFormer | 迁移学习，小数据集预训练+微调 |
| M34 | 生成模型 | VAE/JTVAE, diffusion/flow, scaffold-constrained generation | 分子生成，逆向设计 |
| M35 | 主动学习与代理模型 | Gaussian process, Bayesian optimization, multi-objective BO, active learning | 实验闭环，高效筛选 |
| M36 | 迁移与多任务 | transfer learning, multi-task learning, domain adaptation, few-shot | 跨任务/跨数据域 |

**前沿趋势**:
- **Science 2024**: HTM 逆向设计 = 高通量合成 + Bayesian optimization
- **Nature 2026**: ML 材料发现 + 自动化制造平台的 PSC 闭环框架
- **EES 2025**: 深度学习用于 in-line 成膜监控和性能预测（大面积薄膜 process monitoring）
- **AFM 2025**: ML-guided PSC 实验，2079条数据，R²=0.76，12个验证实验

**本项目已有结果** (AutoGluon, 5-fold CV):
1. WeightedEnsemble_L2 — CV R²: 0.1624
2. RandomForestMSE_BAG_L1 — CV R²: 0.1375
3. ExtraTreesMSE_BAG_L1 — CV R²: 0.1358
4. XGBoost_BAG_L1 — CV R²: 0.0841

**多Agent跨层探索结果** (4-Agent Sequential, fast-only, baseline-as-feature):
- **Best R²**: 0.2962 — L1=agentic_veryloose, L2=F22_maccs (168 features), L3=M31_random_forest, L4=E42_random_split
- **Mean R²**: 0.1086 across 4 unique combinations
- **Key insight**: MACCS fingerprints + RF outperform basic RDKit descriptors (+10% R² improvement over prior best)

---

### Layer 4 — 评估、验证与优化策略 (Evaluation)

| 编号 | 方法类别 | 具体方法 |
|------|----------|----------|
| E41 | 任务类型 | 回归(PCE/band gap/lifetime), 分类(stable/unstable), 排序(top-k) |
| E42 | 数据划分 | random split, **scaffold split**, composition split, **temporal split**, external test |
| E43 | 交叉验证 | 5-fold/10-fold CV, **nested CV**, repeated seeds, leave-one-family-out |
| E44 | 超参搜索 | Optuna/TPE, **Bayesian optimization**, grid/random search, early stopping |
| E45 | 模型可信度 | R²/RMSE/MAE, AUROC/F1/hit rate, **uncertainty**, SHAP/feature importance |

**关键进展**:
- scaffold split 和 temporal split 比 random split 更严格地评估泛化性
- uncertainty quantification（GPR / ensemble / MC dropout）对实验闭环至关重要
- 外部实验验证是PSC领域的金标准（AFM 2025: 12个验证实验）

---

### Layer 5 — 部署、虚拟筛选与自主发现 (Deployment)

| 编号 | 目标 | 说明 |
|------|------|------|
| D51 | 预测目标 | PCE, Voc, Jsc, FF, 稳定性/寿命, band gap, 缺陷钝化, 界面能级, 迁移率 |
| D52 | 虚拟筛选库 | PubChem/ZINC/ChEMBL, Sigma-Aldrich/TCI/Aladdin, Enamine/eMolecules/Mcule |
| D53 | 输出候选 | 分子添加剂/钝化剂, HTM/ETM/SAM, A/B/X组分, 溶剂与工艺配方 |
| D54 | 实验与计算闭环 | DFT验证 → 器件制备与表征 → 稳定性测试 → **新数据回流** |
| D55 | AI Agent / SDL | 自动规划→调用工具→运行→评估→反思→迭代优化 |

**闭环示例**:
```
Agent筛选候选分子 → DFT计算验证 → 合成+器件制备 → 测量PCE → 新数据回流到训练集 → Agent重新训练 → 更好的候选
```

---

## AI Agent 自主探索架构

Agent 不走固定路线（不是MCTS树搜索），而是由 LLM 驱动自主决定探索方向：

| Agent | 职责 | 跨层连接 |
|-------|------|----------|
| **Planner** | 拆分任务与设计搜索空间 | 定义科学目标 |
| **Retriever** | 检索文献、数据库、代码、历史实验 | → S11/S12/S15 (Layer 1) |
| **Feature Agent** | 自动生成 RDKit/DFT/fingerprint/embedding | → F21/F23/F26 (Layer 2) |
| **Model Agent** | 自动尝试 RF/XGBoost/GNN/Uni-Mol/BO | → M31/M33/M35 (Layer 3) |
| **Evaluation Agent** | nested CV, scaffold split, uncertainty, SHAP | → E42/E45 (Layer 4) |
| **Executor / Debugger** | 运行代码，修复报错，记录版本 | 跨所有层 |
| **Experiment / DFT Agent** | 生成 CP2K/Gaussian/实验配方 | → D54 (Layer 5) |
| **Memory** | 记录失败路径、成功路径、新数据回流 | → S17 (Layer 1 闭环) |

### Agent 循环

```
科学目标定义 → Planner拆分 → Retriever搜索数据 → Feature Agent生成特征
→ Model Agent训练模型 → Evaluation Agent评估 → Executor运行/调试
→ Experiment/DFT Agent验证 → Memory记录 → 新数据回流 → 下一轮迭代
```

---

## 过去为什么很难穷举组合？

1. **数据源 × 特征 × 模型 × 评估的组合空间巨大**
2. **DFT, RDKit, 深度模型, 实验数据格式不兼容**
3. **超参数与数据清洗成本高**
4. **小样本 PSC 数据容易过拟合与泄漏**
5. **计算, 实验, 文献三条链路难以闭环**

---

## 文件夹结构

```
hybrid_agent_exploration/
├── AGENTS.md                          # 本文件 — 完整项目参考
├── pipeline_architecture.md           # Mermaid 流程图 + 架构说明
├── literature_survey.md               # 按层分类的文献综述
│
├── configs/
│   ├── search_space.yaml              # 每层候选方法搜索空间定义
│   └── data_sources.yaml              # 数据源连接配置
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                  # 多数据源加载
│   ├── features/                      # Layer 2 实现
│   │   ├── __init__.py
│   │   ├── rdkit_descriptors.py       # F21
│   │   ├── fingerprints.py            # F22: ECFP, KRFP, MACCS, AP, TT
│   │   ├── feature_selector.py        # 特征选择
│   │   ├── perovskite_composition.py  # F24: A/B/X组分特征
│   │   └── learned_representations.py # F26: Uni-Mol/JTVAE embedding
│   ├── models/                        # Layer 3 实现
│   │   ├── __init__.py
│   │   ├── model_registry.py          # 统一模型注册
│   │   ├── traditional_ml.py          # M31: RF, XGBoost, LGBM, CatBoost, SVR, KNN
│   │   ├── pretrained_molecular.py    # M33: Uni-Mol, ChemBERTa wrapper
│   │   ├── active_learning.py         # M35: GP, Bayesian optimization
│   │   └── generative.py              # M34: JTVAE, diffusion
│   ├── evaluation/                    # Layer 4 实现
│   │   ├── __init__.py
│   │   ├── cross_validation.py        # nested CV, scaffold split, temporal split
│   │   ├── hyperparameter.py          # Optuna, Bayesian optimization
│   │   ├── metrics.py                 # R², RMSE, MAE, AUC, F1, uncertainty
│   │   ├── interpretation.py          # SHAP, feature importance
│   │   └── model_comparison.py        # 横向比较
│   └── screening/                     # Layer 5 实现
│       ├── __init__.py
│       ├── virtual_screener.py        # 虚拟筛选管线
│       ├── database_connectors.py     # ZINC, PubChem, Enamine API
│       └── closed_loop.py             # 闭环反馈：新数据回流
│
├── results/
│   ├── exploration_logs/              # Agent 探索日志
│   ├── best_pipelines/                # 最优管线配置 + 模型
│   └── comparison_reports/            # 横向比较报告
│
├── src/
│   ├── cross_layer_sampler.py         # 跨层配置随机采样器
│   ├── worker_agent.py                # 单Agent执行器
│   ├── orchestrator.py                # 多进程并行调度器
│   └── exploration_analyzer.py        # 结果分析与排行榜
│
└── requirements.txt
```

---

## 多Agent并行跨层探索系统

### 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| **Sampler** | `src/cross_layer_sampler.py` | 从 method_registry.yaml 中加权随机采样 L1-L5 方法组合 |
| **Worker** | `src/worker_agent.py` | 单Agent端到端执行：数据加载→特征生成→模型训练→评估→筛选 |
| **Orchestrator** | `src/orchestrator.py` | 多进程并行调度，支持 checkpoint 恢复 |
| **Analyzer** | `src/exploration_analyzer.py` | 排行榜、层贡献分析、Pareto前沿可视化 |

### 运行方式

```bash
# 快速测试（4 agents, 顺序执行）
python src/orchestrator.py --n-agents 4 --max-workers 1 --target delta_pce \
  --baseline-as-feature --fast-only --output results/test_run

# 大规模并行探索（20 agents, 4 workers）
python src/orchestrator.py --n-agents 20 --max-workers 4 --target delta_pce \
  --baseline-as-feature --fast-only --output results/exploration_20

# 使用 convenience script
bash explorations/multi_agent_run.sh
```

### 关键参数

- `--fast-only`: 跳过慢速方法（KRFP 4860-bit 指纹、Optuna 超参搜索、GridSearch）
- `--baseline-as-feature`: 将 `jv_reverse_scan_pce_without_modulator` 作为输入特征（强烈推荐，+188% R²）
- `--max-workers`: 并行进程数；`--max-workers 1` 为顺序执行（适合调试，spawn 开销小）
- `--checkpoint-every N`: 每 N 个 agent 保存一次中间结果

### 已实现的方法覆盖

| Layer | 已实现方法 |
|-------|-----------|
| L1 数据清洗 | agentic_veryloose / agentic_standard / agentic_strict / traditional |
| L2 特征 | F21_rdkit_basic (15-dim), F22_maccs (166-bit), F22_ecfp4/6 (2048-bit), F22_atom_pair (2048-bit), F22_topological_torsion (2048-bit) |
| L3 模型 | M31_random_forest, M31_xgboost, M31_lightgbm, M31_svr, M31_knn |
| L4 评估 | E42_random_split, E43_5fold_cv, E45_shap |
| L5 筛选 | D53_top_k, D54_report_only |

### 技术细节

- **进程模型**: 使用 `multiprocessing.get_context("spawn")` 避免 XGBoost/LightGBM/RDKit 的 fork 死锁
- **spawn 开销**: 每个 worker 首次启动需 ~30-60s 重新导入 heavy libs；建议大批量运行 (>20 agents) 以摊销开销
- **失败隔离**: 单个 agent 失败不会影响其他 agent；错误信息完整记录

---

## 关键文献索引

### 数据来源
- Jacobsson et al., "An open-access database and analysis tool for perovskite solar cells," **Nature Energy**, 2022. (42,400+ 器件, FAIR)
- Perovskite Database Project: https://perovskitedatabase.com
- Materials Project: https://nextgen.materialsproject.org
- NOMAD: https://fairmat-nfdi.github.io/nomad-perovskite-solar-cells-database/

### 特征工程与表征
- "Descriptor Design for Perovskite Material with Compatible GBDT," **JCTC**, 2024.
- Uni-Mol: https://github.com/deepmodeling/Uni-Mol (14/15 SOTA)
- "Transfer learning discovery of molecular modulators for perovskite," arXiv 2511.00204

### ML模型
- "Machine learning for perovskite solar cells: a comprehensive review," **RSC EES**, 2025. (RF/XGBoost最有效)
- "AI for Perovskite Additive Engineering," PMC, 2025.
- Co-PAS: scaffold预筛选 + JTVAE latent + PubChem 25万分子做添加剂筛选

### 闭环与自驱实验
- **Science 2024**: HTM逆向设计 = 高通量合成 + Bayesian optimization
- **Nature 2026**: ML材料发现 + 自动化制造平台的PSC闭环框架
- **EES 2025**: 深度学习用于大面积薄膜 in-line 成膜监控
- **AFM 2025**: ML-guided PSC实验, 2079条数据, R²=0.76, 12个验证实验

### AI Agent
- **SELA**: "Tree-Search Enhanced LLM Agents for AutoML," arXiv 2410.17238, Stanford 2024
- **I-MCTS**: "Enhancing Agentic AutoML via Introspective MCTS," arXiv 2502.14693, EACL 2026
- **ML-Master**: https://github.com/sjtu-sai-agents/ML-Master (上海交大)

### 虚拟筛选
- ZINC: https://zinc.docking.org
- PubChem: https://pubchem.ncbi.nlm.nih.gov
- Enamine REAL: https://enamine.net
- ChEMBL: https://www.ebi.ac.uk/chembl

---

## 关联项目

| 项目 | 路径 | 作用 |
|------|------|------|
| AutoML_EDA (父项目) | `/share/yhm/test/AutoML_EDA/` | EDA基础 + 已有ML结果 |
| Perovskite_Database_Multiagents | `/share/yhm/test/Perovskite_Database_Multiagents/` | 多Agent数据提取管线 (S13) |
| Chemical Merge | `/share/yhm/test/20260216_with_chemical_merge_fast/` | 数据合并上游 |
| Perovskite_Pretrain_Models | 备份中 `desktop/code/` | 预训练模型 |

---

## 验证基准

当前最佳结果 (AutoGluon, 5-fold CV, 特征=basic chemical descriptors):
- **CV R²**: 0.1624 (WeightedEnsemble_L2)
- **Test R²**: 0.1201
- **Test RMSE**: 3.402%

多Agent跨层探索最新基准 (4-Agent, Delta_PCE + baseline feature, fast-only):
- **Best R²**: 0.2962 — agentic_veryloose + MACCS + RandomForest + random_split
- **Mean R²**: 0.1086 | Best RMSE: 2.239%
- **Success rate**: 100% (4/4 agents completed without errors)

AFM 2025 参考基准: R²=0.76 (2079条数据, 12个验证实验)

Agent 探索目标：通过跨层组合优化，显著超过当前基准。
