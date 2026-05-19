# Materials Informatics + AutoML 领域期刊图表标准调研

## 1. 概述

材料信息学（Materials Informatics）结合 AutoML、贝叶斯优化（Bayesian Optimization, BO）和主动学习（Active Learning, AL）的可视化规范，与通用机器学习有显著差异。核心差异在于：

- **物理可解释性**：所有预测必须附带物理单位和实验验证标注
- **不确定性传播**：必须显示误差条、置信区间或预测分布
- **小数据效率**：强调样本效率（sample efficiency）而非大数据性能
- **闭环可视化**：流程图必须体现"计算→实验→反馈"的闭环

## 2. 优化收敛可视化

### 2.1 优化轨迹/收敛曲线（Optimization Trajectories / Convergence Curves）

**常见形式**：
- 折线图：迭代步数（x轴）vs 目标函数值（y轴）
- 多条轨迹对比：不同优化策略（random, BO, AL）的收敛速度比较
- 对数坐标：log(iteration) vs best-so-far objective

**坐标轴规范**：
- x轴："Iteration" 或 "Number of Evaluations"，线性或对数尺度
- y轴：目标函数名称（如 "PCE [%]", "Band Gap [eV]"），必须包含物理单位
- 图例：清晰标注每条曲线的优化策略

**最佳实践**：
- 显示标准差或置信区间的阴影带（shaded confidence band）
- 标注实验验证点（用不同形状标记）
- 对于 AL，标注采集函数选择的下一个实验点
- 使用虚线标注已知最优值（文献基准）

**示例来源**：
- Lookman et al. (2019), *npj Comput. Mater.* — 主动学习收敛曲线的标杆可视化
- Hu et al. (2026), *Science Advances* — Foundation model + AL 的学习曲线

### 2.2 采集函数可视化（Acquisition Function Visualization）

**常见形式**：
- 2D 等高线图：展示采集函数在参数空间的分布
- 1D 切片图：展示单维参数上的采集函数值
- 热图：多参数组合下的采集函数值

**坐标轴规范**：
- x/y轴：参数名称（如 "Annealing Temperature [°C]", "Additive Concentration [mol%]"）
- 颜色条：采集函数值（Expected Improvement, UCB, etc.）
- 标记：已采样点（圆点）、待采样点（星号）、最优预测（叉号）

**最佳实践**：
- 叠加 GP 均值和方差的等高线
- 用不同颜色区分 exploration 和 exploitation 区域
- 对于高维问题，使用 PCA/t-SNE 降维后可视化

## 3. 多目标分析与 Pareto 前沿

### 3.1 Pareto 前沿图（Pareto Front Plot）

**常见形式**：
- 2D 散点图：两个冲突目标的权衡（如 PCE vs Stability）
- 3D 散点图：三个目标的权衡（较少见，通常用 2D 投影）
- 带连接线的散点图：显示 Pareto 最优解的边界

**坐标轴规范**：
- x轴/y轴：目标函数名称，必须包含物理单位和优化方向（max/min）
- 颜色编码：第三个目标或计算成本
- 形状编码：不同算法（MOBO, NSGA-II, random search）

**最佳实践**：
- 用不同透明度标注 dominated vs non-dominated 解
- 显示 hypervolume 指标随迭代的变化
- 标注文献中的实验验证点
- 用箭头指示理想的乌托邦点（utopia point）

**示例来源**：
- Low et al. (2024), *npj Comput. Mater.* — qNEHVI 轨迹的完整多目标评估

### 3.2 Hypervolume 收敛图

**常见形式**：
- 折线图：迭代步数 vs hypervolume 指标
- 多条曲线对比：不同多目标优化算法的 hypervolume 收敛

**坐标轴规范**：
- x轴："Iteration" 或 "Number of Evaluations"
- y轴："Hypervolume Indicator"

## 4. 不确定性量化图

### 4.1 预测不确定性（Predictive Uncertainty）

**常见形式**：
- 带误差棒的散点图：预测值 ± 标准差/置信区间
- 小提琴图：预测分布的展示
- 预测 vs 实际图 + 不确定性带：在 parity plot 上叠加 ±2σ 区域

**坐标轴规范**：
- 与 Parity Plot 一致，但叠加不确定性可视化
- 颜色编码：不确定性大小（高不确定性用暖色，低不确定性用冷色）

**最佳实践**：
- 对于 GPR/ensemble：显示 mean ± 2σ
- 对于 MC Dropout：显示多个 forward pass 的分布
- 标注高不确定性区域（通常对应数据稀疏区域）
- 与实验验证点对比，验证不确定性估计的准确性

### 4.2 校准图（Calibration Plot）

**常见形式**：
- 可靠性图：预测概率 vs 观测频率（分类任务）
- 预测区间覆盖率：实际值落入预测区间的比例 vs 置信水平

## 5. 跨方法比较与基准

### 5.1 学习曲线（Learning Curves）

**常见形式**：
- 双对数坐标：log(training set size) vs log(error metric)
- 多条曲线：训练集和验证集性能随样本量变化
- 不同模型的对比曲线

**坐标轴规范**：
- x轴："Training Set Size" 或 "N_samples"，对数尺度
- y轴：性能指标（R², RMSE, MAE），必须包含物理单位
- 图例：不同模型/特征/策略

**最佳实践**：
- 显示误差棒（多次随机分割的标准差）
- 标注数据效率转折点（diminishing returns）
- 对于小数据 regime，重点展示 N < 1000 的区域
- 与文献基准做水平虚线对比

**示例来源**：
- Hu et al. (2026) — Foundation model + AL 的学习曲线比较

### 5.2 雷达图/蜘蛛图（Radar / Spider Plot）

**常见形式**：
- 多边形雷达图：多个指标（R², RMSE, 训练时间, 可解释性, 稳定性）的综合比较
- 填充区域：不同模型的性能轮廓

**坐标轴规范**：
- 每个轴：一个评估指标，统一归一化到 [0, 1]
- 标注：每个轴的原始数值和方向（越大越好/越小越好）

**最佳实践**：
- 使用半透明填充避免遮挡
- 限制指标数量在 5-7 个以内
- 标注最佳模型的顶点

**示例来源**：
- Xue et al. (2025), *Nature Sci. Rep.* — AL 策略 AUC 雷达图

### 5.3 箱线图/小提琴图（Boxplot / Violin Plot）

**常见形式**：
- 模型间性能分布：不同模型在多次 CV/repeat 中的性能分布
- 特征重要性分布：不同特征子集的稳定性

## 6. 流程与架构图

### 6.1 材料信息学闭环流程图

**常见形式**：
- 块图/流程图："Data → Feature Engineering → Model → Prediction → Experiment → New Data"
- 带反馈箭头的循环图
- 时间线/历史演进图

**设计规范**：
- 使用统一的块形状和颜色编码（数据=蓝色，模型=绿色，实验=橙色，反馈=红色箭头）
- 每个块标注关键工具/方法
- 箭头标注数据流向和转换
- 对于 AutoML，标注自动化决策点

**示例来源**：
- MRS Bulletin (2025) — 历史演进+未来方向的时间线

### 6.2 架构概念图

**常见形式**：
- 分层架构图：展示 L1-L5 各层的组件和连接
- 代理/多代理系统的交互图

## 7. 配色与风格规范

### 7.1 通用规范

- **保守专业**：避免过于鲜艳的配色，优先使用 muted/dark 色调
- **色盲友好**：使用 ColorBrewer 的色盲安全配色（如 Set2, Paired, Dark2）
- **物理单位**：所有坐标轴必须标注物理单位
- **字体大小**：≥ 8pt（Nature 系列要求 Arial/Helvetica ≥ 8pt）
- **分辨率**：TIFF ≥ 600 dpi，EPS/PDF 为矢量格式

### 7.2 材料信息学特有规范

- **实验验证标注**：计算预测点用空心圆，实验验证点用实心星号
- **误差条**：必须包含，不能省略
- **不确定性传播**：用半透明阴影带而非虚线表示置信区间
- **闭环标注**：流程图中用红色粗箭头标注反馈回路

### 7.3 推荐配色方案

| 场景 | 推荐配色 |
|------|---------|
| 单目标优化 | 蓝→绿渐变（viridis） |
| 多目标 Pareto | 不同算法用 Set2/Dark2 |
| 不确定性 | 中心线深色，置信带浅色半透明 |
| 实验验证点 | 红色星号或橙色方块 |
| 失败/异常 | 灰色或暗红色 |

## 8. 附录：典型参考文献中的图表示例

| 论文 | 期刊 | 关键图表类型 | 可借鉴点 |
|------|------|-------------|---------|
| Lookman et al. (2019) | npj Comput. Mater. | 收敛曲线、AL 闭环、SHAP | 实验 AL 的标杆可视化 |
| Low et al. (2024) | npj Comput. Mater. | Pareto front、qNEHVI 轨迹 | 多目标 BO 的完整评估 |
| Hu et al. (2026) | arXiv/Science Advances | 学习曲线、UQ 比较 | Foundation model + AL |
| Xue et al. (2025) | Nature Sci. Rep. | AL 策略 AUC 雷达图 | AutoML + AL benchmark |
| MRS Bulletin (2025) | MRS Bulletin | 流程图、时间线 | 历史演进+未来方向 |

---

> **总结**：材料信息学 + AutoML 领域的图表规范核心在于 **"物理可解释性"** 和 **"不确定性传播"**。与通用 ML 相比，必须强调：物理单位、误差条/置信区间、实验验证标注、以及小数据 regime 下的样本效率。配色应保守专业，流程图需体现闭环，所有预测必须附带不确定性量化。
