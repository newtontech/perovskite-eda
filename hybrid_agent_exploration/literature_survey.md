# Literature Survey — PSC ML 文献综述（按五层管线分类）

> 聚焦钙钛矿太阳能电池（PSC）分子添加剂/组分/工艺优化
> 文献趋势：从"表格数据 + 手工特征 + RF/XGBoost/SVR" → "JTVAE/Uni-Mol 学习表示 + Bayesian/主动学习 + 实验验证/自驱闭环"

---

## Layer 1: 数据来源

### 1.1 钙钛矿公共数据库

**Perovskite Database Project**
- Jacobsson et al., "An open-access database and analysis tool for perovskite solar cells based on the FAIR data principles," *Nature Energy*, 2022. (cited 424+)
- 42,400+ 器件, 每器件100参数, FAIR原则
- https://www.nature.com/articles/s41560-021-00941-3
- https://perovskitedatabase.com

**NREL Perovskite Database**
- 开放获取分析工具, 含bandgap过滤和成分外推
- https://docs.nrel.gov/docs/fy22osti/81870.pdf

**Materials Project**
- Jain et al., "Commentary: The Materials Project," *APL Materials*, 2013.
- DFT计算材料数据库, REST API, 无机钙钛矿结构
- https://nextgen.materialsproject.org

**NOMAD / OQMD / JARVIS**
- NOMAD: FAIR数据存储, https://fairmat-nfdi.github.io/nomad-perovskite-solar-cells-database/
- OQMD: 开放量子材料数据库, https://oqmd.org
- JARVIS: NIST联合自动化存储库, https://jarvis.nist.gov

### 1.2 文献与文本挖掘数据

**Figshare Perovskite Dataset**
- ChemDataExtractor 自动从文献提取, MongoDB/JSON格式, Argonne/DOE资助
- https://figshare.com/articles/dataset/Perovskite_Solar_Cell_Database/13516238

**本项目 Literature Extraction (S13)**
- Perovskite_Database_Multiagents 项目
- PDF → Multimodal LLM (GPT-5-mini) → Crossref → Chemical Data Merge
- 91,357 原始样本 → 5,354 完整案例

### 1.3 化学分子库

- **PubChem**: 1.1亿化合物, https://pubchem.ncbi.nlm.nih.gov
- **ChEMBL**: 生物活性分子数据库, https://www.ebi.ac.uk/chembl
- **ZINC**: 百万级可购买分子, https://zinc.docking.org
- **DrugBank-like molecules**: 药物相关分子

### 1.4 可购买分子库

- **Sigma-Aldrich / TCI / Aladdin / Energy Chemical**: 试剂供应商目录
- **Enamine REAL**: 几十亿虚拟可合成分子, https://enamine.net
- **eMolecules / Mcule**: 供应商聚合, 可购买化合物
- Co-PAS 工作使用了 PubChem 25万分子做添加剂筛选

### 1.5 计算数据

- DFT (VASP/Quantum ESPRESSO/CP2K): HOMO/LUMO/带隙/吸附能/形成能
- SCAPS / drift-diffusion: 器件物理模拟
- High-throughput calculation: 高通量计算筛选

### 1.6 自动化实验数据

- **Science 2024**: 高通量合成 + Bayesian optimization for HTM inverse design
- **Nature 2026**: ML材料发现 + 自动化制造平台的PSC闭环框架
- Self-driving lab logs: 机械臂合成、blade coating/slot-die、in situ PL/imaging

---

## Layer 2: 数据分析与表征

### 2.1 分子描述符 (F21)

- **RDKit**: 200+ 描述符 (分子量、LogP、TPSA、HBD/HBA、QED等)
- **Mordred**: 1800+ 2D/3D描述符
- **PaDEL**: 药效团描述符, 含1D/2D指纹
- "Descriptor Design for Perovskite Material with Compatible GBDT," *JCTC*, 2024. https://pubs.acs.org/doi/10.1021/acs.jctc.4c00465

### 2.2 分子指纹 (F22)

- **ECFP4/ECFP6** (Morgan): 2048-bit, 最常用圆形指纹
- **MACCS Keys**: 166-bit结构键
- **KRFP** (Klekota-Roth): 4860-bit, 生物活性相关亚结构
- Atom Pair / Topological Torsion: 原子对/拓扑扭转指纹

### 2.3 DFT/量化特征 (F23)

- HOMO/LUMO, band gap, dipole, ESP, charge
- Adsorption/formation energy
- 需要量子化学计算，成本高但信息密度大

### 2.4 钙钛矿组分特征 (F24)

- A/B/X site composition
- Ionic radius, tolerance factor, octahedral factor
- Electronegativity, oxidation state
- Gebhardt et al., "Screening for sustainable and lead-free perovskite halide absorbers," *Composites Part B*, 2023.

### 2.5 器件与工艺特征 (F25)

- Architecture: n-i-p / p-i-n
- Transport layers (HTM/ETM/SAM)
- Solvent, annealing temperature/time
- Additive concentration, initial PCE

### 2.6 学习型表示 (F26) — **前沿趋势**

- **JTVAE latent**: Co-PAS 用 scaffold 预筛选 + JTVAE latent 做添加剂筛选
- **Uni-Mol 3D embedding**: 512-dim, 14/15 SOTA, https://github.com/deepmodeling/Uni-Mol
- **Molecular graph embedding**: GNN/MPNN学习
- **Crystal graph**: CGCNN/MEGNet 用于无机钙钛矿
- **SMILES token / SELFIES**: 序列表征, 适合Transformer

### 2.7 图像/谱学表征 (F27) — **新兴方向**

- PL / TRPL / UV-vis: 光致发光光谱
- XRD / GIWAXS: 晶体结构
- SEM / AFM: 表面形貌
- **EES 2025**: in situ PL / reflection 用于大面积薄膜 process monitoring

---

## Layer 3: ML 模型

### 3.1 经典ML (M31)

- **Random Forest**: 多篇PSC文献确认稳定有效
- **XGBoost / LightGBM / CatBoost**: 集成提升方法
  - RSC EES 2025 综述: "RF/XGBoost consistently achieve best performance"
  - PMC 2024: XGBoost预测复合损失, 7-fold CV
- **SVR / KNN / Elastic Net / Ridge / Lasso**: 基线方法

### 3.2 深度学习 (M32)

- **MLP/DNN**: 基础神经网络
- **CNN for images/spectra**: 谱学/图像数据
- **GNN/MPNN**: 分子图神经网络
- **CGCNN/MEGNet**: 晶体图网络, 无机材料

### 3.3 预训练分子模型 (M33) — **前沿**

- **Uni-Mol**: 3D分子预训练, 迁移学习能力强
- **MolCLR**: 分子对比学习
- **ChemBERTa / MolFormer**: 大规模预训练分子Transformer
- 本项目 Perovskite_Database_Multiagents 已集成 UniMol Agent

### 3.4 生成模型 (M34) — **逆向设计**

- **VAE / JTVAE**: 分子潜在空间生成
- **Diffusion / Flow**: 扩散模型/流模型
- **Scaffold-constrained generation**: 骨架约束生成
- Co-PAS: scaffold预筛选 + JTVAE latent

### 3.5 主动学习与代理模型 (M35) — **实验闭环核心**

- **Gaussian process**: 不确定性估计
- **Bayesian optimization**: 高效实验设计
  - Science 2024: HTM逆向设计 = 高通量合成 + BO
- **Multi-objective BO**: 多目标优化 (PCE + 稳定性)
- **Active learning**: 主动选择最有信息量的实验

### 3.6 迁移与多任务 (M36)

- **Transfer learning**: arXiv 2511.00204, 分子描述符→材料性能
- **Multi-task learning**: 同时预测多个性质
- **Domain adaptation**: 跨数据域适应
- **Few-shot**: 低数据学习

### 3.7 AutoML集成

- **AutoGluon**: 本项目已用, WeightedEnsemble_L2, CV R²=0.1624
- "ML-guided optimization of lead-free perovskite solar cells," *PubMed*, 2025.

---

## Layer 4: 评估与验证

### 4.1 任务类型

- 回归: PCE, band gap, lifetime
- 分类: stable/unstable, valid/invalid
- 排序: top-k candidates

### 4.2 数据划分 — **关键**

- Random split: 基础，但可能过拟合
- **Scaffold split**: 按分子骨架划分，更严格的泛化评估
- Composition split: 按组分划分
- **Temporal split**: 按时间划分，模拟真实部署
- External test: 外部独立验证集

### 4.3 交叉验证

- 5-fold / 10-fold CV
- **Nested CV**: 内层选参，外层评估，减少选择偏差
- Repeated seeds: 多随机种子重复
- Leave-one-family-out: 留一族出

### 4.4 超参搜索

- **Optuna / TPE**: 贝叶斯优化，高效
- **Bayesian optimization**: 与主动学习结合
- Grid / random search: 基线

### 4.5 模型可信度

- R², RMSE, MAE (回归)
- AUROC, F1, hit rate (分类)
- **Uncertainty**: GPR / ensemble / MC dropout, 对实验闭环至关重要
- **SHAP / feature importance**: 可解释性
- **Learning curves**: 收敛诊断

### 4.6 外部验证 — **金标准**

- **AFM 2025**: ML-guided PSC实验, 2079条数据, R²=0.76, **12个验证实验**
- DFT验证 → 器件制备 → 表征 → 稳定性测试

---

## Layer 5: 部署与闭环

### 5.1 预测目标

- PCE, Voc, Jsc, FF
- 稳定性/寿命, band gap
- 缺陷钝化效果, 界面能级对齐, 迁移率

### 5.2 虚拟筛选库

- PubChem / ZINC / ChEMBL (公开)
- Sigma-Aldrich / TCI / Aladdin (可购买)
- Enamine / eMolecules / Mcule (可合成)

### 5.3 输出候选

- 分子添加剂 / 钝化剂
- HTM (空穴传输材料) / ETM (电子传输材料) / SAM (自组装单层)
- A/B/X 组分
- 溶剂与工艺配方

### 5.4 闭环 — **最高阶范式**

- DFT验证 → 器件制备与表征 → 稳定性测试 → **新数据回流到训练集**
- **Science 2024**: 高通量合成 + Bayesian optimization for HTM
- **Nature 2026**: ML + 自动化制造平台的PSC闭环框架
- EES 2025: in-line 成膜监控 → 实时反馈

---

## 综述文献

- "Machine learning for perovskite solar cells: a comprehensive review," **RSC EES**, 2025. https://pubs.rsc.org/en/content/articlehtml/2025/el/d5el00041f
- "Key Advancements and Emerging Trends of Perovskite Solar Cells," PMC, 2025.
- "Artificial Intelligence-Based Applications in Perovskite Photovoltaics," ScienceDirect, 2025.
- "Innovating Research Paradigms of Perovskite Solar Cells," Wiley, 2025.
