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
├── reports/                # 分析报告
│   └── eda_report.md       # EDA 报告
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

---

## 🛠️ 技术栈

- **Python 3.11+**
- **Pandas** - 数据处理
- **Matplotlib/Seaborn** - 数据可视化
- **Plotly** - 交互式图表
- **NLTK/Spacy** - 文本分析
- **WordCloud** - 词云生成

---

## 📝 报告

完整的 EDA 报告位于 [`reports/eda_report.md`](reports/eda_report.md)

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
