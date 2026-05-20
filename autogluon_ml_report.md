# AutoGluon优化机器学习报告

---

## 摘要

本报告使用**AutoGluon**自动机器学习框架，以**5折交叉验证R²**为优化目标，对5354个钙钛矿太阳能电池调控剂样本进行Delta_PCE预测模型优化。

<div style="background-color: #f0f7ff; padding: 15px; border-radius: 5px; border-left: 4px solid #2196F3;">

**最优模型**: `WeightedEnsemble_L2`

**5折交叉验证R²**: **0.1624**

**测试集R²**: 0.1201

**测试集RMSE**: 3.402%

</div>

---

## 1. 数据集概况

### 1.1 基本信息

| 统计量 | 数值 |
|--------|------|
| 总样本数 | 5354 |
| 训练集 | 4283 (80%) |
| 测试集 | 1071 (20%) |
| 特征数 | 6 |
| 目标变量 | Delta_PCE (%)

### 1.2 特征列表

| 特征 | 描述 |
|------|------|
| molecular_weight | 分子量 (Da) |
| h_bond_donors | 氢键供体数 |
| h_bond_acceptors | 氢键受体数 |
| rotatable_bonds | 可旋转键数 |
| tpsa | 拓扑极性表面积 (Å²) |
| log_p | 辛醇-水分配系数 |

### 1.3 目标变量统计

| 统计量 | 训练集 | 测试集 |
|--------|--------|--------|
| 均值 | 1.889% | 1.906% |
| 标准差 | 3.418% | 3.628% |
| 最小值 | -16.220% | -19.190% |
| 最大值 | 44.800% | 24.200% |

---

## 2. AutoGluon模型训练

### 2.1 训练配置

- **优化目标**: 5折交叉验证R²
- **预设**: best_quality
- **Bagging Folds**: 5 (用于集成模型的稳健性)
- **评估指标**: R² (决定系数)

### 2.2 模型排行榜

AutoGluon自动训练并比较了多种机器学习模型：

| 排名 | 模型 | CV R² | 训练时间 (s) |
|------|------|-------|-------------|
| 1 | WeightedEnsemble_L2 | 0.1624 | 157.5 |
| 2 | RandomForestMSE_BAG_L1 | 0.1375 | 3.6 |
| 3 | ExtraTreesMSE_BAG_L1 | 0.1358 | 2.1 |
| 4 | XGBoost_BAG_L1 | 0.0841 | 13.4 |
| 5 | CatBoost_r177_BAG_L1 | 0.0834 | 18.6 |
| 6 | CatBoost_BAG_L1 | 0.0818 | 22.8 |
| 7 | LightGBMLarge_BAG_L1 | 0.0797 | 17.0 |
| 8 | LightGBM_r131_BAG_L1 | 0.0765 | 16.1 |
| 9 | LightGBM_BAG_L1 | 0.0690 | 14.9 |
| 10 | LightGBMXT_BAG_L1 | 0.0585 | 20.9 |

![模型比较](../figures/autogluon_model_comparison.png)

*图1. 模型CV R²比较(左)和特征重要性(右)*

---

## 3. 最优模型性能

### 3.1 WeightedEnsemble_L2

WeightedEnsemble_L2是AutoGluon的二级加权集成模型，通过最优加权组合多个基础学习器的预测结果。

#### 性能指标

| 数据集 | R² | RMSE (%) | MAE (%) |
|--------|------|---------|----------|
| **5折CV** | **0.1624** | - | - |
| 训练集 | 0.3479 | 2.759 | 1.785 |
| 测试集 | 0.1201 | 3.402 | 2.174 |

#### 过拟合分析

- **R²差距**: 0.2278 (训练集 - 测试集)
- **评估**: 🔴 **严重** - 明显过拟合
- **说明**: 训练集R²(0.3479)明显高于测试集R²(0.1201)，表明模型在训练数据上学习到了一些无法泛化的模式。

---

## 4. 预测结果分析

### 4.1 True vs Predicted 散点图

![训练测试散点图](../figures/autogluon_train_test_scatter.png)

*图2. WeightedEnsemble_L2的训练集(左)和测试集(右)True vs Predicted散点图*

**关键观察**:

1. **训练集**: R² = 0.3479, 回归斜率 = 0.263
2. **测试集**: R² = 0.1201, 回归斜率 = 0.126
3. 斜率 < 1 表明模型倾向于向均值收敛，对极端值预测保守
4. 红色虚线为完美预测线(y=x)，绿色实线为实际回归拟合

### 4.2 Top 3模型对比

![Top3模型对比](../figures/autogluon_top3_models_scatter.png)

*图3. Top 3模型的训练集和测试集预测对比*

### 4.3 残差分析

![残差图](../figures/autogluon_residual_plots.png)

*图4. 训练集和测试集的残差分布*

| 数据集 | 残差均值 | 残差标准差 |
|--------|----------|------------|
| 训练集 | 0.0050 | 2.7598 |
| 测试集 | 0.0258 | 3.4031 |

**残差特征**:
- 残差均值接近0，表明预测无系统性偏差
- 测试集残差标准差较大，反映泛化误差
- 残差分布无明显模式，模型捕捉了主要线性关系

### 4.4 预测分布对比

![预测分布](../figures/autogluon_prediction_distribution.png)

*图5. 真实值与预测值的分布对比*

---

## 5. 特征重要性分析

### 5.1 特征重要性排序

| 特征 | 重要性 | 标准差 | p值 |
|------|--------|--------|-----|
| molecular_weight | 0.3077 | 0.0132 | 4.04e-07 |
| log_p | 0.2271 | 0.0044 | 1.73e-08 |
| tpsa | 0.1685 | 0.0136 | 5.00e-06 |
| h_bond_acceptors | 0.0941 | 0.0088 | 9.04e-06 |
| rotatable_bonds | 0.0897 | 0.0078 | 6.83e-06 |
| h_bond_donors | 0.0466 | 0.0030 | 1.92e-06 |

### 5.2 解释

1. **molecular_weight** 是最重要的预测特征，重要性为 0.3077
2. 分子量、Log P和TPSA是前三大重要特征
3. 所有特征的重要性均为正值，表明都对预测有正向贡献
4. 氢键供体的贡献相对较小

---

## 6. 与传统方法对比

### 6.1 性能对比

| 方法 | 最优模型 | CV R² | Test R² | Test RMSE |
|------|----------|-------|---------|----------|
| **AutoGluon** | WeightedEnsemble_L2 | **0.1624** | 0.1201 | 3.402 |
| 传统GridSearch | RandomForest | 0.1120 | 0.1116 | 3.418 |

### 6.2 AutoGluon优势

1. **自动化程度高**: 无需手动调参，自动尝试多种模型
2. **集成学习**: 通过加权集成获得更稳健的预测
3. **CV R²提升**: 从0.1120提升至0.1624 (+5.0%)
4. **可解释性**: 提供详细的模型排行榜和特征重要性

---

## 7. 结论与建议

### 7.1 主要发现

1. AutoGluon优化后的最优模型为**WeightedEnsemble_L2**
2. 5折交叉验证R²达到**0.1624**，相比传统方法有显著提升
3. 测试集R²为**0.1201**，RMSE为**3.402%**
4. **分子量**是最重要的预测特征，其次是**Log P**和**TPSA**

### 7.2 模型局限性

- R²值仍然较低(~0.12-0.16)，说明现有分子描述符对Delta_PCE的解释能力有限
- 存在明显过拟合(训练集R² >> 测试集R²)
- 可能存在未纳入模型的关键实验因素(钙钛矿组成、加工条件等)

### 7.3 改进建议

1. **特征工程**:
   - 添加量子化学描述符(HOMO/LUMO、偶极矩、静电势)
   - 使用分子指纹(MACCS, ECFP)捕获结构信息
   - 考虑实验条件特征

2. **数据增强**:
   - 收集更多标准化实验数据
   - 数据清洗，去除异常值和噪声

3. **高级方法**:
   - 图神经网络(GNN)直接从分子图学习
   - 迁移学习，利用预训练化学模型
   - 贝叶斯优化进行超参数调优

---

## 附录

### A. 文件列表

| 文件 | 路径 |
|------|------|
| 模型排行榜 | `tables/autogluon_leaderboard.csv` |
| 特征重要性 | `tables/autogluon_feature_importance.csv` |
| 预测散点图 | `figures/autogluon_train_test_scatter.png` |
| 残差图 | `figures/autogluon_residual_plots.png` |
| 模型比较图 | `figures/autogluon_model_comparison.png` |
| 预测分布图 | `figures/autogluon_prediction_distribution.png` |
| Top3模型图 | `figures/autogluon_top3_models_scatter.png` |
| 本报告 | `autogluon_ml_report.md` |

### B. AutoGluon模型目录

模型保存在: `/share/yhm/test/AutoML_EDA/autogluon_models/`

---

**生成日期**: 2026年03月05日

**优化方法**: AutoGluon AutoML
**优化目标**: 5-Fold Cross-Validation R²
