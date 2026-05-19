# Molecular ML 领域期刊图表标准调研

> 调研范围：JCTC, JCIM, Digital Discovery (RSC), Nature Computational Science, Nature Machine Intelligence, Chemical Science
> 调研日期：2026-05-07
> 聚焦：分子机器学习（Molecular Machine Learning）论文中高频出现的图表类型与可视化最佳实践

---

## 1. 概述

分子机器学习（Molecular ML）领域的可视化图表具有鲜明的学科特色：一方面需要精确呈现化学结构信息，另一方面需要表达高维数据的统计规律与模型行为。通过分析 JCTC、JCIM、Digital Discovery、Nature Computational Science、Nature Machine Intelligence 和 Chemical Science 等高水平期刊近年来的分子 ML 论文，我们归纳出以下 **8 种核心图表类型**及其规范。

**核心发现：**
- **RDKit** 是 2D 分子结构绘图的事实标准（de facto standard），几乎被所有期刊接受
- **化学空间降维图**（t-SNE/UMAP/PCA）是展示分子分布的必备图表
- **SHAP + RDKit 联动** 已成为分子可解释性分析的主流范式
- **Parity plot（预测-真实值散点图）** 是模型性能评估的标配
- **Diverging colormap** 是分子归因可视化的首选配色方案

---

## 2. 分子结构绘图规范

### 2.1 2D 分子结构图

**使用工具：** RDKit (`rdkit.Chem.Draw`)

**在期刊中的典型应用：**
- 展示数据集中的代表性分子
- 高亮显示关键子结构或官能团
- SHAP/归因值的热力图叠加
- 骨架树（Scaffold Tree）可视化

**RDKit 关键绘图参数：**
```python
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# SVG 输出（推荐，矢量图）
d2d = rdMolDraw2D.MolDraw2DSVG(350, 300)

# 关键选项
d2d.drawOptions().addAtomIndices = True       # 原子编号
d2d.drawOptions().addBondIndices = True       # 键编号
d2d.drawOptions().addStereoAnnotation = True  # R/S, E/Z 标记
d2d.drawOptions().atomLabels = {0: "R₁"}      # 自定义标签
d2d.drawOptions().annotationFontScale = 0.75  # 注释字体缩放
```

**绘图细节规范：**

| 元素 | 推荐做法 | 说明 |
|------|----------|------|
| **原子标签** | 默认元素符号（C 不显示） | RDKit 默认隐藏碳和氢 |
| **杂原子颜色** | 遵循 CPK 配色：O=红, N=蓝, S=黄, Cl=绿, P=橙 | 国际通用标准 |
| **键型** | 单键/双键/三键线型区分，芳香键用虚线或圆圈内虚线 | RDKit 自动处理 |
| **高亮** | `highlightAtoms` + `highlightBonds` + `highlightAtomColors` | 支持多色叠加 |
| **立体化学** | 楔形键（实线/虚线）表示 | `addStereoAnnotation=True` 添加 R/S 标记 |
| **输出格式** | SVG（矢量）> PNG（位图，≥300 dpi） | 期刊通常要求 ≥600 dpi TIFF |

**多分子并排展示：**
- 使用 `Draw.MolsToGridImage()` 生成网格布局
- 每个分子下方标注 ID 或属性值
- 统一缩放确保键长一致：`drawOptions().fixedBondLength = True`

### 2.2 3D 分子结构图

**使用工具：**
- **Py3Dmol**（Jupyter 内嵌，基于 3Dmol.js）
- **NGLView**（交互式，支持轨迹）
- **RDKit + Py3Dmol 联动**

**RDKit → Py3Dmol 工作流：**
```python
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
AllChem.MMFFOptimizeMolecule(mol)

view = py3Dmol.view(data=Chem.MolToMolBlock(mol),
                    style={"stick": {}, "sphere": {"scale": 0.3}})
view.zoomTo()
```

**3D 可视化规范：**

| 场景 | 推荐样式 | 备注 |
|------|----------|------|
| 静态结构展示 | `stick` 或 `sphere` | 小分子用 stick，大分子用 cartoon |
| 构象比较 | 多构象叠加，高亮显示差异原子 | 使用 `opacity` 控制透明度 |
| 分子动力学轨迹 | `line` + 动画播放 | NGLView 支持 TRR/XTC/DCD 格式 |
| 药效团特征 | RDKit `ChemicalFeatures` + 3D 标记 | 氢键供体/受体/芳香中心 |

**期刊要求：**
- Digital Discovery (RSC) 建议：如包含复杂 3D 分子，提供独立下载文件和嵌入 PDF 两种形式
- 静态论文图片建议导出高分辨率 PNG（≥600 dpi）

---

## 3. 化学空间可视化

### 3.1 降维方法对比与选择

化学空间可视化是将高维分子表征（指纹、描述符）投影到 2D/3D 空间的核心手段。

| 方法 | 全局结构保留 | 局部邻域保留 | 计算效率 | 适用场景 |
|------|-------------|-------------|----------|----------|
| **PCA** | ★★★★★ | ★★☆☆☆ | ★★★★★ | 快速概览、线性可分数据 |
| **t-SNE** | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ | 聚类展示、中小数据集 (<10k) |
| **UMAP** | ★★★★☆ | ★★★★★ | ★★★★☆ | 大规模数据、保留全局+局部结构 |
| **GTM** | ★★★☆☆ | ★★★★★ | ★★★☆☆ | 概率密度估计、不确定性量化 |

*数据来源：PMC11733715 (2022) 对 59 个 ChEMBL 子集的系统性评估*

### 3.2 t-SNE / UMAP 化学空间图

**典型布局：**
- **散点图（Scatter plot）**：每个点代表一个分子
- **Hexbin 图**：高密度区域用六边形网格聚合
- **等高线图**：展示密度分布

**标注规范：**
```python
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# UMAP 降维（推荐参数）
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='jaccard')
embedding = reducer.fit_transform(fingerprints)

# 绘图
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                     c=property_values, cmap='viridis',
                     s=30, alpha=0.6, edgecolors='none')
plt.colorbar(scatter, label='Property Value (e.g., pIC50)')
ax.set_xlabel('UMAP-1', fontsize=12)
ax.set_ylabel('UMAP-2', fontsize=12)
ax.set_title('Chemical Space Colored by Bioactivity')
```

**关键规范：**

| 元素 | 推荐做法 |
|------|----------|
| **颜色编码** | 分子属性（活性、LogP、毒性）或类别标签 |
| **点大小** | 可映射至分子量或另一属性 |
| **透明度** | `alpha=0.5-0.7`，避免重叠遮挡 |
| **颜色条** | 必须添加，标注单位和范围 |
| **代表分子** | 在图上直接嵌入关键分子的 2D 结构（inset） |
| **聚类标注** | 用椭圆或凸包（convex hull）圈出主要簇 |

**t-SNE 超参数（JCTC/JCIM 常见设置）：**
- `perplexity`: 5-50（小数据集用较小值）
- `learning_rate`: "auto"（sklearn ≥1.2）
- `n_iter`: 1000
- `metric`: "euclidean"（描述符）或 "jaccard"（指纹）

**UMAP 超参数：**
- `n_neighbors`: 5-50（平衡局部/全局结构）
- `min_dist`: 0.0-0.5（控制簇间距离）
- `metric`: "euclidean", "cosine", "correlation"

### 3.3 PCA Biplot（双标图）

同时展示样本点和特征向量的经典方法：
- 箭头表示描述符对主成分的贡献方向
- 箭头长度表示贡献大小
- 用于解释化学空间的物理化学含义

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 标准化后 PCA
pca = PCA(n_components=2)
scores = pca.fit_transform(StandardScaler().fit_transform(descriptors))

# Biplot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(scores[:, 0], scores[:, 1], alpha=0.5)

# 绘制特征向量
features = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA']
for i, feature in enumerate(features):
    ax.arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3,
             head_width=0.05, color='red')
    ax.text(pca.components_[0, i]*3.5, pca.components_[1, i]*3.5, feature)
```

---

## 4. 可解释性分析图

### 4.1 SHAP 分子归因分析

这是分子 ML 领域最具特色的可视化类型之一，将 SHAP 值映射到分子结构上。

**主流实现方式：**

#### 方式 A：Riniker & Landrum 方法（基于 Morgan 指纹）
将指纹位的 SHAP 值映射回中心原子：
```python
from rdkit import Chem
from rdkit.Chem import AllChem
import shap

# 计算 Morgan 指纹位对原子的映射
mol = Chem.MolFromSmiles(smiles)
bi = {}
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=bi)

# 聚合 SHAP 值到原子
atom_shap = defaultdict(float)
for bit, shap_val in zip(active_bits, shap_values):
    for atom_idx, _ in bi[bit]:
        atom_shap[atom_idx] += shap_val
```

#### 方式 B：Fragment-level Shapley（FragShapley）
在片段级别计算 Shapley 值，再映射到原子/键：
- 使用 **diverging colormap**（如 `RdBu_r`, `PiYG`, `BrBG`）
- 红色/暖色 = 正向贡献（增加预测值）
- 蓝色/冷色 = 负向贡献（降低预测值）
- 颜色强度 = 贡献绝对值大小

**RDKit 热力图绘制：**
```python
from rdkit.Chem.Draw import rdMolDraw2D
from collections import defaultdict

# 准备原子颜色
atom_colors = {}
for atom_idx, shap_val in atom_shap.items():
    # 归一化到 [-1, 1]
    norm_val = max(-1, min(1, shap_val / max_abs_shap))
    if norm_val > 0:
        atom_colors[atom_idx] = (1.0, 1.0 - norm_val, 1.0 - norm_val)  # 红色渐变
    else:
        atom_colors[atom_idx] = (1.0 + norm_val, 1.0 + norm_val, 1.0)  # 蓝色渐变

# 绘制
d2d = rdMolDraw2D.MolDraw2DCairo(400, 350)
d2d.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()),
                 highlightAtomColors=atom_colors,
                 highlightBonds=[], highlightBondColors={})
d2d.FinishDrawing()
```

**XSMILES 等交互式工具的高级特性：**
- 在分子图上叠加半透明热力层
- 颜色盲友好调色板（ColorBrewer `RdYlBu`）
- 支持自定义 colormap domain（突出特定范围的归因值）

### 4.2 GNN 可解释性图

**常见方法：**

| 方法 | 可视化形式 | 颜色方案 |
|------|-----------|----------|
| **GNNExplainer** | 子图掩码（重要节点/边保留） | Sequential（重要性强度） |
| **GradCAM / CAM** | 节点热力图 | Diverging（正负贡献） |
| **Attention Maps** | 边权重可视化 | Sequential（注意力强度） |
| **SubgraphX** | 关键子图提取 | 原子颜色标注 |

**GNNExplainer 可视化规范：**
```python
from torch_geometric.explain import Explainer, GNNExplainer

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(mode='regression', task_level='graph', return_type='raw')
)

explanation = explainer(x, edge_index)
# explanation.visualize_graph()  # PyG 内置可视化
```

**期刊中的典型展示：**
- 原始分子图 vs. 解释子图并排
- 热力图叠加在 2D 分子结构上
- AUC 评估归因质量（Ground truth vs. CAM weights）

### 4.3 SHAP Summary / Force Plot

**全局特征重要性（模型级）：**
```python
import shap
import matplotlib.pyplot as plt

# Summary plot（描述符级别）
shap.summary_plot(shap_values, X, feature_names=descriptor_names,
                  max_display=20, show=False)
plt.title("Top 20 Molecular Descriptors by SHAP Importance")
```

**Cumulative SHAP 分析（JCIM 常用）：**
- 展示累积 SHAP 贡献达到 50%/75%/90% 所需的特征数
- MACCS keys 通常需 ~20 个特征达到 90%
- RDKit 描述符通常需 ~55-100 个特征
- ECFP4 通常需 ~400 个特征

---

## 5. 分子相似性网络与聚类图

### 5.1 分子相似性网络（Similarity Network）

**构建方法：**
- 节点 = 分子
- 边 = Tanimoto 相似度 ≥ 阈值（通常 0.3-0.7）
- 边权重 = 相似度值

**工具链：** RDKit（指纹）+ NetworkX（图）+ PyVis/Matplotlib（可视化）

```python
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import networkx as nx

# 计算 Morgan 指纹
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]

# 构建网络
G = nx.Graph()
for i in range(len(mols)):
    for j in range(i+1, len(mols)):
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        if sim >= 0.3:
            G.add_edge(i, j, weight=sim)

# 使用力导向布局
pos = nx.spring_layout(G, k=0.5, iterations=50)
```

**可视化规范：**

| 元素 | 推荐做法 |
|------|----------|
| **节点颜色** | 聚类标签或活性类别 |
| **节点大小** | 度中心性（连接数）或分子量 |
| **边粗细** | Tanimoto 相似度（越粗越相似） |
| **布局算法** | Force-directed（spring_layout, Kamada-Kawai） |
| **代表分子** | 每个簇选 1-2 个代表性分子显示 2D 结构 |

### 5.2 层次聚类与树状图（Dendrogram）

**应用场景：**
- 骨架层次分类（Scaffold Tree）
- 分子库聚类分析
- 活性类别层次关系

**RDKit + SciPy 实现：**
```python
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# 计算距离矩阵（1 - Tanimoto）
dist_matrix = []
for i in range(len(fps)):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dist_matrix.extend([1-s for s in sims])

# Ward 层次聚类
Z = linkage(dist_matrix, method='ward')

# 绘制树状图
fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, labels=labels, leaf_rotation=90, ax=ax)
ax.set_ylabel('Ward Distance', fontsize=12)
ax.set_title('Hierarchical Clustering of Molecular Library')
```

**最佳实践：**
- 树状图 + 热图组合（Clustered Heatmap）
- 行/列按聚类结果重排
- 热图单元格 = Tanimoto 相似度矩阵
- 颜色：暖色（红/黄）= 高相似，冷色（蓝）= 低相似

### 5.3 骨架树（Scaffold Tree）

**工具：** RDKit `MurckoScaffold` + 自定义层次结构

**规范：**
- 根节点 = 最简化骨架
- 分支 = 逐步添加环系或侧链
- 节点颜色 = 平均活性值或分子数量
- 节点大小 = 对应分子数

---

## 6. 分子属性分布图

### 6.1 单变量分布

**常用图表类型：**

| 图表类型 | 适用场景 | 工具 |
|----------|----------|------|
| **直方图（Histogram）** | 单属性分布概览 | `matplotlib`, `seaborn` |
| **核密度估计（KDE）** | 平滑分布曲线 | `seaborn.kdeplot` |
| **箱线图（Boxplot）** | 多组比较 | `seaborn.boxplot` |
| **小提琴图（Violin plot）** | 分布形状 + 统计量 | `seaborn.violinplot` |
| **累积分布（CDF/ECDF）** | 阈值分析 | `seaborn.ecdfplot` |

**小提琴图规范（JCIM/Chemical Science 常见）：**
```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
properties = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds']

for ax, prop in zip(axes.flat, properties):
    sns.violinplot(data=df, x='Class', y=prop, ax=ax, palette='Set2')
    ax.set_title(prop, fontsize=11)
    ax.set_xlabel('')

plt.suptitle('Molecular Property Distributions by Class', fontsize=14)
plt.tight_layout()
```

**关键规范：**
- 箱线图内嵌于小提琴图中（展示 Q1/Q2/Q3）
- 散点叠加显示异常值
- 颜色按类别区分（active/inactive, 或不同数据集）
- 标注 Lipinski/Veber 规则阈值线（如 MW=500, LogP=5）

### 6.2 多变量分布

**雷达图（Radar/Spider Plot）：**
- 展示单个分子或分子集的多维属性
- 常见维度：MW, LogP, TPSA, HBD, HBA, RotBonds, QED
- 归一化到 [0, 1] 或原始值

**散点矩阵（Pair Plot）：**
```python
sns.pairplot(df[['MolWt', 'LogP', 'TPSA', 'HBD', 'Activity']],
             hue='Activity', corner=True, diag_kind='kde')
```

### 6.3 化学空间覆盖度比较

**典型场景：** 比较训练集/测试集/虚拟筛选库的分布重叠

**实现方式：**
- 2D PCA/UMAP 密度等高线图叠加
- 直方图并排对比
- KDE 曲线重叠比较
- 统计检验：Kolmogorov-Smirnov 检验

---

## 7. 模型性能与基准对比图

### 7.1 Parity Plot（预测-真实值散点图）

这是分子 ML 论文中最普遍的模型评估图。

**标准布局：**
```python
fig, ax = plt.subplots(figsize=(6, 6))

# 散点
ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='none', label='Test Set')

# 1:1 对角线
lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
ax.plot(lim, lim, 'k--', lw=1, label='Ideal')

# 性能指标文本框
textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

ax.set_xlabel('Experimental Value', fontsize=12)
ax.set_ylabel('Predicted Value', fontsize=12)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(loc='lower right')
```

**规范要点：**

| 元素 | 推荐做法 |
|------|----------|
| **对角线** | 1:1 虚线（`k--`），必须包含 |
| **轴范围** | 相等（`set_aspect('equal')`） |
| **颜色区分** | 训练集（蓝）/ 测试集（红）/ CV（绿） |
| **误差指标** | R², RMSE, MAE 标注在图内文本框 |
| **透明度** | `alpha=0.5-0.7`，展示点密度 |
| **边缘线** | 无（`edgecolors='none'`） |

### 7.2 残差图（Residual Plot）

```python
fig, ax = plt.subplots(figsize=(6, 4))
residuals = y_true - y_pred
ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='none')
ax.axhline(y=0, color='k', linestyle='--', lw=1)
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Residual')
ax.set_title('Residual Plot')
```

### 7.3 模型对比图

**柱状图/箱线图对比：**
- x 轴：不同模型（RF, XGBoost, GNN, etc.）
- y 轴：性能指标（R², RMSE, MAE）
- 误差条：标准差（来自 cross-validation folds）
- 颜色：按模型类别区分（树模型=绿，NN=蓝，GP=橙）

**Taylor Diagram（高级）：**
- 同时展示相关系数、标准差、RMSE
- 适用于多模型综合比较

### 7.4 ROC / PR 曲线（分类任务）

```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
```

---

## 8. 分子生成与优化轨迹图

### 8.1 生成/优化轨迹

**典型展示：**
- x 轴：优化步数 / 生成轮次
- y 轴：目标属性值（如 QED, pIC50, 合成可及性 SA）
- 多条轨迹线：不同初始分子或不同算法
- 散点大小：分子复杂度或多样性

**散点轨迹图：**
```python
fig, ax = plt.subplots(figsize=(10, 6))
for traj_id, traj in enumerate(trajectories):
    steps = range(len(traj))
    ax.plot(steps, [m['property'] for m in traj],
            alpha=0.6, label=f'Trajectory {traj_id+1}')
    # 标注起点和终点分子
    ax.scatter(0, traj[0]['property'], s=100, marker='o', c='green', zorder=5)
    ax.scatter(len(traj)-1, traj[-1]['property'], s=100, marker='*', c='red', zorder=5)

ax.set_xlabel('Optimization Step')
ax.set_ylabel('Target Property Value')
ax.set_title('Molecular Optimization Trajectories')
ax.legend()
```

### 8.2 生成分子集属性分布对比

**关键比较维度：**
- 生成分子 vs. 训练集：MW, LogP, TPSA, QED 分布
- 内部多样性（Internal Diversity）
- 新颖性（Novelty）
- 合成可及性（SAscore）

**常用图表：**
- 并排小提琴图/直方图
- 累积分布函数（CDF）比较
- MMD（最大均值差异）数值标注

---

## 9. 配色与风格规范

### 9.1 期刊通用要求

**Digital Discovery (RSC) 图像规范：**
- 格式：TIFF（≥600 dpi），或 EPS/PDF（编辑部会转换）
- 尺寸：单栏 8.3 cm，双栏 17.1 cm，最大高度 23.3 cm
- 彩色图片：在线和印刷均免费使用
- 文字/数值/比例尺必须清晰可读

**Nature 系列通用规范：**
- 首次提交接受高质量 PDF/PNG
- 接收后需提交 EPS/AI 格式
- 避免使用 3D 效果、阴影、渐变填充（非数据驱动）
- 字体：Arial 或 Helvetica，≥8 pt

### 9.2 分子可视化配色方案

**原子颜色（CPK 标准）：**

| 元素 | 颜色 | RGB |
|------|------|-----|
| C | 灰色 | (128, 128, 128) |
| H | 白色 | (255, 255, 255) |
| N | 蓝色 | (48, 80, 248) |
| O | 红色 | (255, 13, 13) |
| S | 黄色 | (255, 255, 48) |
| P | 橙色 | (255, 165, 0) |
| Cl | 绿色 | (31, 240, 31) |
| Fe | 深橙 | (224, 102, 51) |

**归因/热力图配色：**

| 场景 | 推荐 Colormap | 说明 |
|------|--------------|------|
| 正负贡献 | `RdBu_r`, `BrBG`, `PiYG` | Diverging，零值居中为白色/浅色 |
| 仅正贡献 | `YlOrRd`, `Reds`, `viridis` | Sequential，低→高 |
| 注意力权重 | `YlGnBu`, `Blues` | Sequential |
| 类别标签 | `Set2`, `Paired`, `tab10` | Qualitative，色盲友好 |

**ColorBrewer 推荐（色盲友好）：**
- Sequential: `Blues`, `Greens`, `Oranges`, `viridis`
- Diverging: `RdBu`, `BrBG`, `PRGn`, `RdYlBu`
- Qualitative: `Set1`, `Set2`, `Paired`, `Dark2`

**避免使用的配色：**
- `jet` / `rainbow`：感知不均匀，色盲不友好
- 红-绿组合（色盲难以区分）
- 纯蓝-红文本组合（色差模糊）

### 9.3 色盲友好设计

- 使用 **ColorBrewer** 或 **Viridis** 系列 colormap
- 除颜色外，同时使用形状、大小、纹理区分数据
- 在图中直接标注数值或类别名称
- 打印前转换为灰度检查可读性

### 9.4 推荐 Python 配色工具

```python
# ColorBrewer
import matplotlib.pyplot as plt
plt.cm.get_cmap('RdBu_r')

# 科学配色 (Crameri et al.)
# 安装: pip install cmcrameri
from cmcrameri import cm
plt.cm.get_cmap('cm.batlow')   # 色盲友好 sequential
plt.cm.get_cmap('cm.vik')      # 色盲友好 diverging

# 检查色盲友好性
# 安装: pip install colorblindly
```

---

## 10. 总结与推荐工具链

### 10.1 推荐工具链

| 图表类型 | 首选工具 | 辅助工具 |
|----------|----------|----------|
| 2D 分子结构 | RDKit | ChemDraw (手动调整) |
| 3D 分子结构 | Py3Dmol / NGLView | PyMOL, ChimeraX |
| 化学空间图 | UMAP (Python) + Matplotlib | t-SNE, PCA |
| SHAP 归因 | SHAP + RDKit | XSMILES (交互式) |
| GNN 解释 | PyG Explain + RDKit | GNNExplainer |
| 相似性网络 | RDKit + NetworkX + PyVis | Cytoscape |
| 聚类/树状图 | SciPy + Matplotlib/Seaborn | Scaffold Hunter |
| 属性分布 | Seaborn (violin/box/kde) | Plotly (交互式) |
| 性能对比 | Matplotlib + Seaborn | - |
| 生成轨迹 | Matplotlib | Plotly |

### 10.2 期刊投稿检查清单

- [ ] 分子结构使用 RDKit 绘制，遵循 CPK 颜色标准
- [ ] 所有图片分辨率 ≥ 600 dpi（印刷）或矢量格式
- [ ] 颜色方案色盲友好，避免 jet/rainbow
- [ ] 散点图包含颜色条（colorbar）和图例
- [ ] Parity plot 包含 1:1 对角线和性能指标
- [ ] 化学空间图标注代表分子或关键簇
- [ ] SHAP 图使用 diverging colormap，标注颜色含义
- [ ] 图内文字 ≥ 8 pt，清晰可见
- [ ] 3D 结构图提供独立下载文件（如 Digital Discovery 要求）
- [ ] 所有图表自包含（不依赖正文即可理解）

---

## 参考文献与资源

1. **RDKit Documentation**: https://www.rdkit.org/docs/
2. **XSMILES (JCIM 2023)**: https://doi.org/10.1186/s13321-022-00673-w
3. **FragShapley (ChemRxiv)**: https://doi.org/10.26434/chemrxiv.15002302
4. **GNN Explainability Survey**: https://doi.org/10.1007/s10618-022-00870-z
5. **UMAP for Chemical Space**: https://doi.org/10.1101/2025.07.03.663077
6. **Dimensionality Reduction Comparison (PMC11733715)**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11733715/
7. **SHAP in JCIM**: https://doi.org/10.1021/acs.jcim.5c02015
8. **Color Palette Guidelines (PMC7561171)**: https://pmc.ncbi.nlm.nih.gov/articles/PMC7561171/
9. **Molecular Coloring Best Practices (PMC9377702)**: https://pmc.ncbi.nlm.nih.gov/articles/PMC9377702/
10. **Digital Discovery Author Guidelines**: https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/digital-discovery
11. **ColorBrewer**: https://colorbrewer2.org/
12. **Scientific Colormaps (Crameri)**: https://www.fabiocrameri.ch/colourmaps/

---

> 本调研基于对 2022-2026 年间 JCTC, JCIM, Digital Discovery, Nature Computational Science, Nature Machine Intelligence, Chemical Science 期刊中分子 ML 相关论文的系统性分析，结合 RDKit、PyG、SHAP 等工具文档整理而成。
