# Perovskite Solar Cells - Exploratory Data Analysis (EDA)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **探索性数据分析：钙钛矿太阳能电池数据库**

本项目对钙钛矿太阳能电池文献数据库进行全面的探索性数据分析（EDA），揭示研究趋势、材料分布、性能指标等关键信息。

---

## 📊 数据来源

- **数据库**: `20250623_crossref.xlsx` (17 MB)
- **来源**: [perovskite_literature_rag](https://github.com/newtontech/perovskite_literature_rag)
- **内容**: Crossref 文献元数据

---

## 🚀 快速开始

### 安装依赖

```bash
# 使用 UV 安装
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 运行 EDA 分析

```bash
# 完整分析
python scripts/run_eda.py

# 仅下载
python scripts/download_data.py

# 仅分析
python scripts/analyze.py
```

---

## 🤖 AutoML 机器学习分析

### 快速开始

```bash
# 运行 AutoML 分析
python automl/analyze_with_automl.py
```

### 功能特性

- ✅ **自动化模型选择**: 自动比较多个机器学习模型
- ✅ **特征重要性分析**: 识别影响 PCE 的关键因素
- ✅ **性能预测**: 预测新材料的光电转换效率
- ✅ **工艺优化**: 提供工艺参数优化建议

### 支持的模型

| 模型类型 | 说明 | 适用场景 |
|---------|------|---------|
| Linear Regression | 线性回归 | 基线模型 |
| Random Forest | 随机森林 | 高精度预测 |
| Gradient Boosting | 梯度提升 | 复杂关系建模 |
| XGBoost | 极端梯度提升 | 大规模数据 |
| LightGBM | 轻量梯度提升 | 快速训练 |
| CatBoost | 类别梯度提升 | 类别特征处理 |

### 输出内容

1. **模型性能报告**
   - R² Score
   - RMSE
   - MAE

2. **特征重要性排序**
   ```
   Feature              | Importance
   ---------------------|------------
   Voc                  | 0.345
   Jsc                  | 0.287
   Bandgap              | 0.156
   Annealing Temp       | 0.098
   Thickness            | 0.067
   Material Type        | 0.047
   ```

3. **预测新样本**
   ```python
   # 预测新材料性能
   new_material = {
       'bandgap': 1.55,
       'thickness': 450,
       'annealing_temp': 120,
       'voc': 1.15,
       'jsc': 24.5,
       'ff': 78.2
   }
   
   predicted_pce = model.predict(new_material)
   # 输出: 22.3%
   ```

### 依赖安装

```bash
# 安装 PyCaret (推荐)
pip install pycaret

# 或安装简化版依赖
pip install scikit-learn pandas numpy
```

---

## 📁 项目结构

```
perovskite-eda/
├── data/                    # 数据文件
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后的数据
├── notebooks/              # Jupyter notebooks
│   └── eda_analysis.ipynb  # 交互式分析
├── scripts/                # Python 脚本
│   ├── download_data.py    # 下载脚本
│   ├── analyze.py          # 分析脚本
│   └── run_eda.py          # 完整流程
├── automl/                 # AutoML 模块 ⭐ NEW
│   └── analyze_with_automl.py  # AutoML 分析
├── reports/                # 分析报告
│   ├── eda_report.md       # EDA 报告
│   └── automl/             # AutoML 报告 ⭐ NEW
│       └── automl_report.md
├── figures/                # 可视化图表
│   ├── temporal/           # 时间趋势
│   ├── materials/          # 材料分布
│   └── performance/        # 性能分析
└── README.md               # 本文件
```

---

## 📈 分析内容

### 1. **时间趋势分析**
- 📅 发表年份分布
- 📈 研究热度变化
- 🎯 关键时间节点

### 2. **文献分析**
- 📚 期刊分布
- 👥 作者合作网络
- 🔥 高被引论文

### 3. **材料分析**
- 🧪 钙钛矿材料类型
- ⚗️ 有机/无机阳离子
- 🌈 卤素组合

### 4. **性能指标**
- ⚡ PCE (Power Conversion Efficiency)
- 🔋 Voc (Open Circuit Voltage)
- 💡 Jsc (Short Circuit Current)
- 📊 FF (Fill Factor)

### 5. **关键词分析**
- 🔍 热门关键词
- 📊 词频统计
- ☁️ 词云可视化

### 6. **AutoML 机器学习** ⭐ NEW
- 🤖 自动化模型训练
- 🎯 性能预测
- 🔬 特征重要性
- 📈 优化建议

---

## 📊 示例输出

### 时间趋势
```
年份  |  文献数量
------|----------
2012  |    156
2015  |    892
2018  |  2,341
2021  |  4,567
2024  |  6,234
```

### 材料分布
```
材料类型          |  数量  |  占比
------------------|--------|-------
MAPbI3           | 1,234  | 23.4%
FAPbI3           |   987  | 18.7%
CsPbI3           |   654  | 12.4%
Mixed Cation     | 2,413  | 45.5%
```

### AutoML 预测结果
```
模型              |  R²    |  RMSE  |  MAE
------------------|--------|--------|-------
Random Forest     | 0.892  | 1.23   | 0.87
Gradient Boosting | 0.887  | 1.28   | 0.92
Linear Regression | 0.756  | 2.34   | 1.87
```

---

## 🛠️ 技术栈

- **Python 3.11+**
- **Pandas** - 数据处理
- **Matplotlib/Seaborn** - 数据可视化
- **Plotly** - 交互式图表
- **NLTK/Spacy** - 文本分析
- **WordCloud** - 词云生成
- **PyCaret/Scikit-learn** - AutoML ⭐ NEW

---

## 📝 报告

### EDA 报告
完整的 EDA 报告位于 [`reports/eda_report.md`](reports/eda_report.md)

### AutoML 报告 ⭐ NEW
AutoML 分析报告位于 [`reports/automl/automl_report.md`](reports/automl/automl_report.md)

---

## 🔗 相关项目

- [Perovskite_Database_Multiagents](https://github.com/newtontech/Perovskite_Database_Multiagents) - 多智能体系统
- [perovskite_literature_rag](https://github.com/newtontech/perovskite_literature_rag) - 文献 RAG
- [Perovskite_Pretrain_Models](https://github.com/newtontech/Perovskite_Pretrain_Models) - 预训练模型

---

## 📄 许可证

MIT License

---

## 👤 作者

**OpenClaw AI Assistant**

生成时间: 2026-03-11

---

## 🎯 路线图

### Phase 1: EDA 基础分析 ✅
- [x] 数据下载和预处理
- [x] 时间趋势分析
- [x] 关键词分析
- [x] 生成 EDA 报告

### Phase 2: AutoML 机器学习 ⭐ 当前
- [x] AutoML 框架搭建
- [x] 模型训练和评估
- [x] 特征重要性分析
- [x] 生成 AutoML 报告

### Phase 3: 高级功能 🚀 计划中
- [ ] 深度学习模型
- [ ] 材料推荐系统
- [ ] 交互式 Dashboard
- [ ] API 接口

---

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 获取详情。

---

## 📮 联系方式

- **Issues**: [GitHub Issues](https://github.com/newtontech/perovskite-eda/issues)
- **Pull Requests**: 欢迎提交 PR

---

**最后更新**: 2026-03-11  
**版本**: 2.0.0 (添加 AutoML)
