# PSC + ML 领域期刊图表标准调研

> 调研范围: Nature Energy, Joule, EES, AFM, ACS Energy Letters, Advanced Materials 等顶刊中 ML-for-PSC 论文的常见图表类型与规范
> 调研日期: 2026-05-07

---

## 1. 概述

钙钛矿太阳能电池（PSC）与机器学习交叉领域的论文在图表呈现上呈现以下特点：

- **模型验证导向**: 由于 ML 预测的可信度是核心关切，parity plot（预测 vs 实际）几乎成为标配
- **可解释性驱动**: SHAP 分析图在 2023–2025 年间迅速成为主流，远超传统的 feature importance bar plot
- **不确定性量化**: Gaussian Process Regression (GPR) 和 ensemble 方法的置信区间图在涉及实验闭环（closed-loop）的论文中频繁出现
- **结构可视化**: 钙钛矿晶体结构（ABX₃）和有机添加剂分子结构的化学图示用于连接 ML 结果与物理直觉
- **SI 扩展**: 几乎所有模型诊断图（残差分布、学习曲线、超参搜索）均置于 Supporting Information

---

## 2. 图表类型详细分析

### 2.1 Feature Importance Visualization（特征重要性图）

**常见形式:**
- **水平条形图 (Horizontal Bar Plot)**: 最经典形式，按重要性排序展示 Top-N 特征
- **SHAP Summary Plot (Beeswarm/Dot)**: 2023 年后成为顶刊首选，展示每个特征的 SHAP 值分布与方向
- **Violin Plot**: 展示 SHAP 值的分布密度，反映特征影响的变异性

**坐标轴规范:**
- X轴: "Mean |SHAP value|" 或 "Feature importance (%)" / "Average SHAP value"
- Y轴: 特征名称，通常按重要性降序排列，Top 10–20 个
- 单位: 若目标为 PCE，SHAP 值单位常标注为 "SHAP value (%)" 或绝对值单位

**颜色方案:**
- SHAP Summary Plot: 使用 **红-蓝双色渐变**（red = high feature value, blue = low feature value）或 **黄-紫渐变**（yellow = high, purple = low）
- 传统 Bar Plot: 单色（深蓝、墨绿或灰色），或使用渐变色强调排名
- 顶刊偏好低饱和度的学术配色，避免荧光色

**示例来源:**
- *Materials* 2025: "Average SHAP values from four machine learning algorithms using one-hot feature encoding, illustrating the top 20 features influencing PCE"
- *AIP Advances* 2024: SHAP summary plot 用黄-紫渐变表示 feature value 高低
- *PMC/Feature Selection Review* 2023: SHAP scatter plot of AR_A and its SHAP value

**SI 扩展:**
- 完整特征重要性列表（All features）通常放在 Table Sx
- 不同模型（RF/XGBoost/LightGBM）的 SHAP 对比放在 Figure Sx

---

### 2.2 Parity Plot (Predicted vs. Actual / 预测值 vs 实际值)

**常见形式:**
- **散点图 (Scatter Plot)**: 训练集和测试集用不同颜色/形状区分（如蓝色圆点 = train, 橙色圆点 = test）
- **1:1 参考线**: 虚线或实线表示 y = x（perfect prediction），必不可少
- **置信/误差带**: 部分高影响力论文会添加 ±10% 或 ±1 RMSE 的误差带

**坐标轴规范:**
- X轴: "Observed / Experimental PCE (%)" 或 "Actual value"
- Y轴: "Predicted PCE (%)" 或 "Predicted value"
- 轴范围: 通常保持 x 和 y 轴量程一致，确保 1:1 线呈 45°
- 标注: 在图内或图注中明确标注 R², RMSE, MAE，格式如 "R² = 0.86, RMSE = 2.08%"

**颜色方案:**
- Train/Test 区分: 蓝色系（train）vs 橙色系（test）为最常见组合
- 数据密度高时采用半透明（alpha = 0.5–0.7）避免重叠
- 部分论文用颜色编码第三个维度（如 bandgap 或年份）

**示例来源:**
- *Advanced Intelligent Systems* 2025: "The parity plot in Figure 2c shows an R² = 0.80"
- *Wiley* (Gok et al.): 预测 Eg vs 实验 Eg，虚线为 0 error line
- *Nature Energy* (Jacobsson et al., 2022) 相关工作: 预测值与实验值的高度一致性散点图
- *J. Appl. Cryst.* (Process Informatics): "Parity plot for the trained model predictions vs calculated bandgap testing values"

**SI 扩展:**
- 不同目标变量（PCE, Voc, Jsc, FF）的 parity plot 常放在 Figure S3–S6
- 5-fold CV 每一折的 parity plot 用于展示稳定性

---

### 2.3 SHAP Dependence & Interaction Plot（SHAP 依赖与交互图）

**常见形式:**
- **SHAP Dependence Plot**: 单特征的 SHAP 值 vs 该特征原始值，展示非线性关系
- **SHAP Interaction Plot**: 用颜色编码第二个特征，展示交互效应
- **SHAP Summary Plot (Beeswarm)**: 全局展示所有特征的 SHAP 分布

**坐标轴规范:**
- X轴: 特征原始值（如 "FA composition", "Annealing temperature (°C)", "Molecular weight (g/mol)"）
- Y轴: "SHAP value" 或 "Impact on model output"
- 0线: 水平参考线标注，表示无影响

**颜色方案:**
- 单特征依赖图: 灰度或单色散点，加 LOWESS/平滑曲线
- 交互图: 颜色条（colorbar）表示交互特征的值，常用 viridis 或 coolwarm
- Beeswarm: 特征值高-低渐变（红蓝或黄紫）

**示例来源:**
- *Materials* 2025: "SHAP value distribution for FA and Cs composition. Point color encodes SHAP contribution to predicted PCE (warm = positive, cool = negative). The yellow bands denote the recommended composition ranges."
- *PMC/SCAPS Data* 2025: "SHAP value dependency plot" 展示非线性依赖
- *AIP Advances* 2024: "positive/negative SHAP value indicates that the input property with the given value (indicated by color: yellow for high values and purple for low values) positively/negatively affects the output variable (Voc)"

**SI 扩展:**
- 所有特征的 dependence plot 集合放在 Figure S7–S10
- 关键交互作用的补充分析在 SI text 中说明

---

### 2.4 Uncertainty Quantification（不确定性量化图）

**常见形式:**
- **GPR 预测带 (Prediction Interval)**: 蓝色实线为均值预测，浅蓝色阴影为 ±1σ/±2σ 置信区间
- **Ensemble 误差棒**: 多次随机种子或 bagging 预测的标准差用误差棒表示
- **MC Dropout 不确定性**: 深度学习论文中，用散点+误差棒或半透明散布表示预测方差

**坐标轴规范:**
- X轴: 实验观测值或样本索引
- Y轴: 预测值或残差
- 图注必须说明不确定性的统计定义（如 "Shaded region represents ±1 standard deviation from the GP posterior"）

**颜色方案:**
- 预测均值: 深蓝/黑色实线
- 置信区间: 同色系的浅色调填充（light blue, light orange），透明度 0.2–0.4
- 训练点: 红色或黑色散点

**示例来源:**
- *Science Advances* 2025 (Peptide materials): "The dotted lines denote β sheet assembly... Parity plot between the ML-predicted and the experimentally measured IR ratio"
- *JMI* 2025 (Photonic curing): BO-GP heat maps with parity plots showing predicted vs experimentally determined values
- *DTU Thesis* (ML Quantum Mechanics): "GPR provides estimates of the prediction uncertainties... blue lines and light blue areas"
- *Nature Reviews Materials / npj* 2019 (Active Learning): "Gaussian process (GP) or any machine learned model... adaptive sampling using uncertainties"

**SI 扩展:**
- 不同核函数（RBF, Matern）的 GPR 不确定性对比
- Acquisition function (UCB, EI) 的演化图放在 SI

---

### 2.5 Chemical Space / Molecular Fingerprint Visualization（化学空间可视化）

**常见形式:**
- **PCA / t-SNE / UMAP 降维散点图**: 将分子指纹（ECFP, MACCS）降维到 2D/3D
- **化学空间覆盖图**: 训练集 vs 虚拟筛选库（PubChem/ZINC）在降维空间的分布
- **元素表示可视化**: 无机钙钛矿领域，将元素 embedding 用 PCA 投影到 2D

**坐标轴规范:**
- X/Y轴: "PC1", "PC2" 或 "t-SNE 1", "t-SNE 2"，标注解释方差比例（如 "PC1 (34%)"）
- 3D 图: 标注 "PC1", "PC2", "PC3"，视角（azimuth/elevation）在图注中说明
- 颜色/形状编码: 必须在图例中明确（如 target property, compound class, train/test/external）

**颜色方案:**
- 连续变量（PCE, bandgap）: viridis, plasma, coolwarm 渐变
- 分类变量（stable/unstable, train/test）: 高对比度离散色（蓝/橙/绿）
- 顶刊偏好避免 jet/rainbow，多用 perceptually uniform colormaps

**示例来源:**
- *Physical Review Letters* 2018 (CGCNN): "Visualization of the element representations learned from the perovskite dataset... projection on a 2D plane using principal component analysis"
- *ACS Chem. Rev.* 2022: "(t-SNE, UMAP), and the models therefore can be simplified... perovskite materials in photocatalysis- and photovoltaics"
- *JCTC* 2024 / Co-PAS: JTVAE latent space 可视化用于分子生成

**SI 扩展:**
- 不同降维方法（PCA vs t-SNE vs UMAP）的对比图
- 不同指纹（ECFP vs MACCS）的化学空间对比
- 3D 旋转视角的交互式/多视角静态图

---

### 2.6 Model Comparison / Benchmark（模型比较与基准图）

**常见形式:**
- **柱状图 (Bar Chart)**: 比较不同模型的 R², RMSE, MAE，常按性能排序
- **Radar Plot (Spider Plot)**: 多指标综合评价（R², RMSE, MAPE, Training Time）
- **箱线图/小提琴图 (Boxplot/Violin Plot)**: 展示 k-fold CV 中各模型的性能分布稳定性
- **表格 + 热力图**: 模型 × 指标矩阵，数值 cell 着色

**坐标轴规范:**
- Bar Chart X轴: 模型名称（RF, XGBoost, SVR, NN, GP 等）
- Bar Chart Y轴: 指标名称及单位（如 "R²", "RMSE (%)", "MAE (eV)"）
- 误差棒: 必须标注是标准差（SD）还是标准误（SE），基于几次 CV

**颜色方案:**
- 每个模型固定一种颜色，全文一致（如 RF = 绿, XGBoost = 蓝, NN = 红）
- 渐变色柱: 从深到浅表示性能优劣
- 避免使用超过 6–7 种颜色，超过时采用同色系深浅区分

**示例来源:**
- *Energies* 2024: "Table 2. Prediction errors for each model's efficiency" + 图4a 的 predicted vs actual
- *Advanced Energy and Sustainability Research* (Open Source Pipeline): "Model Performances" 表格，比较 XGBoost/NN/GP 的 RMSE 和 R²
- *Springer* (ML Photodetectors Review): "Radar plot of R² index of different ML algorithms; Bar chart of RMSE index for different ML algorithms"
- *RSC* (D3DD00171G): "CC-plots for the linear model(left, R²=0.58) and random forest(right, R²=0.98)" 并排对比

**SI 扩展:**
- 完整模型超参数表（Table S3）
- 各模型在所有 k-folds 上的详细性能（Table S4）
- 训练时间/推理时间对比（Figure S8）

---

### 2.7 Residual Analysis（残差分析图）

**常见形式:**
- **残差 vs 拟合值散点图 (Residual vs Fitted)**: 检查异方差性和模式
- **残差直方图 (Residual Histogram)**: 检验正态分布，常叠加核密度估计（KDE）曲线
- **残差箱线图 (Residual Boxplot)**: 按目标变量区间分组的残差分布
- **Q-Q Plot**: 残差分位数 vs 理论正态分位数

**坐标轴规范:**
- Residual vs Fitted: X = "Predicted value" 或 "Fitted value"; Y = "Residual" 或 "Error"
- Histogram: X = "Residual"; Y = "Frequency" 或 "Density"
- 必须包含 y = 0 参考线

**颜色方案:**
- 散点: 半透明蓝/灰色（alpha = 0.5）
- 直方图: 浅蓝填充 + 深蓝边框，KDE 用不同颜色实线
- 多模型对比: 每个模型一种颜色，半透明叠加

**示例来源:**
- *Advanced Energy and Sustainability Research* 2024 (Open Source Pipeline): "Residual analysis by calculating the differences between the observed and predicted values... Boxplots showing the residuals for different ranges of true J_sc"
- *DTU Thesis* (ML Quantum Mechanics): "Histograms of the prediction residuals of the train and test set... distributed evenly around 0 eV"
- *Performance Enhancement of Sn-based PSC* 2026: "Residual plot indicating random error distribution of random forest predictions"
- *auditor R package* (方法论参考): "Boxplots of absolute values of residuals. Dots are in similar places, hence RMSE for both models is almost identical"

**SI 扩展:**
- 所有目标变量的残差分析（Figure S4–S6）
- 不同数据清洗策略后的残差对比

---

### 2.8 Molecular / Crystal Structure Diagrams（分子与晶体结构图）

**常见形式:**
- **2D 化学结构式**: 有机添加剂、钝化剂、HTM/ETM 的分子结构
- **3D 晶体结构图**: ABX₃ 钙钛矿八面体（BX₆）和 A 位阳离子，常用 VESTA/CrystalMaker 风格
- **器件结构示意图**: n-i-p / p-i-n 层状结构，标注各层材料
- **决策树可视化**: 用于解释 tree-based model 的决策路径

**坐标轴规范:**
- 结构图通常无坐标轴，但需标注:
  - 键长、晶格参数（如 a = 6.3 Å）
  - 原子颜色图例（Pb = 灰, I = 紫, C = 黑, N = 蓝 等）
  - 晶面指数（如 (001), (110)）

**颜色方案:**
- 遵循 CPK 颜色规范或材料学期刊惯例
- 同一元素在全文中保持同色
- 背景白色，避免复杂纹理

**示例来源:**
- *Clemson Thesis* (Lead-free oxides): "Candidate elements featuring the chemical space, the yellow elements occupy the A sites, and the red elements occupy the B sites"
- *AIP Advances* 2024: 有机阳离子结构图（MePEAI, MePEACl 等）与 SHAP 结果关联
- *PR Letters* 2018 (CGCNN): "The perovskite structure type. Visualization of the two principal dimensions with principal component analysis"
- *ChemDataExtractor/Figshare datasets*: 晶体结构示意图作为数据描述的一部分

**SI 扩展:**
- 所有候选分子的 2D 结构（Figure S11–S20）
- DFT 优化后的 3D 坐标（CIF 文件）

---

### 2.9 Learning Curve（学习曲线）

**常见形式:**
- **Log-log 坐标图**: 测试集 MAE/RMSE vs 训练集大小
- **双曲线图**: 训练误差 + 验证误差随样本量或迭代次数的变化
- **收敛图**: Bayesian Optimization 中 acquisition function 或目标值的迭代演化

**坐标轴规范:**
- X轴: "Number of training data" 或 "Number of iterations"，log 尺度常见
- Y轴: "MAE (eV/atom)" 或 "RMSE (%)"，log 尺度常见
- 标注: 与 benchmark 方法的数值对比

**颜色方案:**
- 训练/验证线: 蓝/橙或黑/红实线
- 阴影带: 表示 k-fold 的标准差
- 参考线: 灰色虚线表示 DFT 计算精度或理论极限

**示例来源:**
- *Physical Review Letters* 2018 (CGCNN): "The learning curve in Fig. 2 shows a straight line in log-log scale, indicating a steady increase of prediction performance as the number of training data increases"
- *DTU Thesis*: "Learning curve for the ML model showing validation MAE as a function of the number of materials/states in the training set"
- *Data Brief* 2025 (SCAPS): "Grid search plot with 5-fold cross-validation: solid line – validation set RMSE, dashed line – train set RMSE"

**SI 扩展:**
- 不同模型架构的学习曲线对比
- 不同特征集的学习曲线

---

## 3. 配色与风格规范

### 3.1 顶刊通用配色趋势

| 配色类型 | 常见使用场景 | 具体色值参考 |
|---------|------------|------------|
| **蓝-橙互补** | Train/Test 区分、模型对比 | `#1f77b4` (蓝) / `#ff7f0e` (橙) |
| **红-蓝发散** | SHAP value 高低、正负影响 | `#d62728` (红) / `#2c3e50→#3498db` (蓝) |
| **Viridis/Plasma** | 连续变量（PCE, bandgap, 概率） | matplotlib `viridis`, `plasma` |
| **Coolwarm** | 零-centered 数据（SHAP, 残差） | matplotlib `coolwarm` |
| **灰度 + 强调色** | 基线 + 最优模型突出 | 灰色系 + 单点高饱和红/绿 |

### 3.2 字体与尺寸规范

- **正文字体**: Arial, Helvetica 或 Times New Roman, 7–9 pt
- **坐标轴标签**: 8–10 pt，带单位（如 "PCE (%)"）
- **图注 (Caption)**: 顶刊通常要求独立提交，但正文图中字体需保证缩小后可读
- **多面板标注**: (a), (b), (c), (d) 使用粗体，位置统一在左上角或左下角

### 3.3 线条与标记规范

- **1:1 参考线**: 灰色虚线（`linestyle='--'`, `alpha=0.7`），线宽 1.0–1.2 pt
- **散点标记**: 圆形（`o`），大小 20–50，半透明（`alpha=0.5–0.7`）
- **误差棒**: 两端带帽（`capsize=3–5`），线宽 1.0 pt
- **拟合曲线**: 实线或 LOWESS 平滑，线宽 1.5–2.0 pt

---

## 4. SI 图表规范

### 4.1 SI 中常见图表类型

| 图表内容 | 是否常见于 SI | 说明 |
|---------|------------|------|
| 完整 parity plots (所有 folds) | ✅ 高频 | Figure S3–S6 |
| 完整 feature importance (所有特征) | ✅ 高频 | Figure S7 / Table S2 |
| 残差分析 (所有模型/所有目标) | ✅ 高频 | Figure S8–S10 |
| 学习曲线 | ✅ 中频 | Figure S11 |
| 超参数搜索热图/曲线 | ✅ 中频 | Figure S12 / Table S3 |
| 化学空间 3D 视角/对比 | ✅ 中频 | Figure S13–S15 |
| 数据分布/EDA 直方图 | ✅ 高频 | Figure S1–S2 |
| 分子结构全集 | ✅ 中频 | Figure S16–S20 |
| 模型架构细节 (NN/GNN) | ⚠️ 低频 | 部分期刊要求 |

### 4.2 SI 图表标注规范

- 编号: **Figure S1, Figure S2, ...** 或 **Fig. S1, Fig. S2**
- 表格: **Table S1, Table S2, ...**
- 图注需完整独立理解，不依赖正文（因 SI 常被单独阅读）
- 统计指标: 必须注明样本数（N）、交叉验证折数（k）、随机种子

---

## 5. 不推荐的做法

基于顶刊趋势和审稿常见意见，以下做法应尽量避免：

1. **Rainbow/Jet  colormap**: 在连续变量可视化中，彩虹色会引入视觉伪影，已被 *Nature* 系列明确建议避免
2. **无单位坐标轴**: "Predicted value" 而不标注单位（如 %, eV, mA cm⁻²）会被要求修改
3. **训练集和测试集不区分**: Parity plot 中不标注 train/test 或颜色不分，难以评估过拟合
4. **SHAP 图不解释颜色**: 仅放 beeswarm 图而不在图注说明 "red = high feature value" 会导致读者困惑
5. **残差不分析**: 仅放 parity plot 而不分析残差分布，会被质疑模型偏差（bias）和异方差性
6. **缺失 1:1 参考线**: Parity plot 中遗漏 y = x 线被视为不规范
7. **过多特征无排序**: Feature importance 图若特征数 > 15 且不排序，可读性极差
8. **3D 图无视角说明**: 化学空间 3D 散点图若未注明视角或不可旋转，信息传达效率低
9. **混淆 R 和 R²**: 图注中需明确是 Pearson R 还是 coefficient of determination R²
10. **SI 中缺少原始数据**: 越来越多期刊要求提供生成图表的源数据（Source Data），应在 SI 中准备

---

## 6. 参考文献与示例论文索引

| 论文 | 期刊/年份 | 关键图表类型 |
|------|----------|------------|
| Jacobsson et al. | *Nature Energy*, 2022 | 数据库结构、统计分布图 |
| Lu et al. (CGCNN) | *PRL*, 2018 | Learning curve, PCA element embedding |
| Gok et al. | *Wiley*, 2022 | Parity plot, PCE vs bandgap |
| Jiang et al. (GPR/Active Learning) | *Nature Energy* 相关, 2022 | BO convergence, parity plot |
| Li et al. (Two-model strategy) | *Wiley*, 2021 | Parity plot, PCE map, bandgap map |
| Open Source Pipeline | *Advanced Energy and Sustainability Research*, 2024 | Residual boxplot, model comparison table, histogram |
| SHAP + Stacking Ensemble | *Materials*, 2025 | SHAP summary, SHAP dependence, parity plot |
| SCAPS-1D + CatBoost | *Data Brief*, 2025 | SHAP summary, feature importance bar, grid search plot |
| ML for Surface Modifiers | *AIP Advances*, 2024 | SHAP summary (黄-紫), correlation matrix |
| Feature Selection Review | *PMC*, 2023 | SHAP scatter, last-place elimination workflow |

---

*本文件为 PSC + ML 领域图表制作的设计参考，建议在生成论文图表前对照检查。*
