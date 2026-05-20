# Perovskite EDA 📊

探索性数据分析项目 - 钙钛矿太阳能电池性能预测

## 🎯 项目目标

1. 对钙钛矿太阳能电池数据库进行全面 EDA 分析
2. 构建高性能预测模型（R² > 0.85）
3. 识别影响性能的关键因素
4. 生成可复现的研究报告

## 📊 数据来源

### Perovskite Database

本项目使用 [Perovskite Database](https://github.com/Jesperkemist/perovskitedatabase_data) 作为数据源。

- **数据文件**: `data/raw/perovskite_database_all.csv`
- **数据字典**: `data/data_dictionary.md`
- **版本信息**: `data/data_version.txt`
- **记录数**: ~50,000+
- **特征数**: 65+
- **时间范围**: 2012-2024

### 数据描述

钙钛矿太阳能电池性能数据库包含：
- 器件结构和组成
- 钙钛矿材料特性
- 性能指标（PCE, Voc, Jsc, FF）
- 稳定性数据
- 化学性质
- 文献信息

## 🚀 快速开始

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 下载数据

```bash
# 下载 Perovskite Database
wget -O data/raw/perovskite_database_all.csv \
  https://github.com/Jesperkemist/perovskitedatabase_data/raw/main/Perovskite_database_content_all_data.csv
```

### 运行数据概览

```bash
python scripts/data_overview.py
```

### 生成规范研究包

根目录 `Makefile` 提供规范入口，输出默认写入已忽略的
`hybrid_agent_exploration/results/verified_discovery_runs/` 目录：

```bash
make research-package
make research-package-pdf
make test-research-package
```

常用环境变量：

- `SOURCE_TABLE`: 输入 CSV/XLSX 源表，默认使用本项目 QSPR 合并表。
- `DATASET_ID`: 稳定数据集或运行标识，默认 `canonical-research-package`。
- `ARTIFACT_DIR`: 研究包输出目录，默认 `hybrid_agent_exploration/results/verified_discovery_runs/$(DATASET_ID)`。
- `EVIDENCE_MODE`: 证据模式，支持 `external-cached` 或 `source-columns`。
- `INPUT_SCOPE`: 输入范围声明，支持 `selected-subset` 或 `full-source`。
- `MIN_VERIFIED_ROWS`: 最少验证行数。
- `TOP_K`: 候选排序输出数量。
- `CANDIDATE_SOURCE`: 可选外部候选源 CSV/XLSX。
- `CANDIDATE_SOURCE_NAME`: 可选外部候选源名称。

示例：

```bash
SOURCE_TABLE=/path/to/source.xlsx \
DATASET_ID=jpcl-full-source \
EVIDENCE_MODE=external-cached \
INPUT_SCOPE=full-source \
MIN_VERIFIED_ROWS=10 \
TOP_K=100 \
make research-package
```

快速冒烟运行可使用：

```bash
SOURCE_TABLE=/path/to/source.csv make research-package-smoke
```

`research-package-pdf` 会把已生成的 `main_text_report.md` 和
`supporting_information.md` 转换为 PDF；如果未安装 `pandoc`，该目标会直接失败并提示安装。

### 运行 EDA 分析

```bash
python src/eda/analyze.py
```

### 训练模型

```bash
python src/automl/train.py
```

## 📁 项目结构

```
perovskite-eda/
├── data/
│   ├── raw/                    # 原始数据
│   │   └── perovskite_database_all.csv
│   ├── processed/              # 处理后的数据
│   ├── data_dictionary.md      # 数据字典
│   └── data_version.txt        # 版本信息
├── scripts/
│   ├── data_overview.py        # 数据概览脚本
│   └── download_data.sh        # 数据下载脚本
├── src/
│   ├── eda/
│   │   └── analyze.py          # EDA 分析脚本
│   └── automl/
│       └── train.py            # AutoML 训练脚本
├── reports/
│   └── eda_report.md           # EDA 报告
├── docs/
│   └── METHODOLOGY.md          # 方法论文档
├── tests/                      # 测试
├── requirements.txt            # 依赖
└── README.md                   # 本文件
```

## 📈 分析内容

### 1. 数据探索
- 数据概览和统计
- 缺失值分析
- 异常值检测
- 数据分布可视化

### 2. 特征工程
- 特征选择
- 特征转换
- 特征重要性分析

### 3. 模型训练
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### 4. 模型评估
- R² 分数（目标 > 0.85）
- MAE（目标 < 1.5%）
- 特征重要性排序
- 残差分析

## 📊 预期成果

1. **EDA 报告**: 完整的探索性数据分析报告
2. **训练模型**: R² > 0.85 的预测模型
3. **特征重要性**: Top 20 关键特征
4. **可视化图表**: 10+ 高质量图表
5. **可复现代码**: 完整的分析脚本

## 🔧 技术栈

- **Python**: 3.11+
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn, plotly
- **机器学习**: scikit-learn, xgboost, lightgbm
- **AutoML**: autogluon (可选)

## 📝 开发进度

### Phase 1: 数据准备 ✅
- [x] 数据下载和存储
- [x] 数据字典创建
- [x] 数据概览脚本

### Phase 2: EDA 分析 ⏳
- [ ] 单变量分析
- [ ] 双变量分析
- [ ] 多变量分析
- [ ] 时间序列分析

### Phase 3: 模型训练 ⏳
- [ ] 数据预处理
- [ ] 特征工程
- [ ] 模型训练
- [ ] 模型评估

## 🤝 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

MIT License

## 📞 联系方式

- **项目维护者**: NewtonTech Team
- **GitHub**: https://github.com/newtontech/perovskite-eda

---

**最后更新**: 2026-03-12
