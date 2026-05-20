# Legacy Workflow（历史流程）

> ⚠️ 仅历史参考，不保证可运行。

本页保留早期 README 中的旧流程与旧目录描述，用于追溯历史上下文。

## 旧版快速开始（已过时）

### 安装依赖（历史）

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 下载数据（历史）

```bash
wget -O data/raw/perovskite_database_all.csv \
  https://github.com/Jesperkemist/perovskitedatabase_data/raw/main/Perovskite_database_content_all_data.csv
```

### 旧脚本入口（历史）

```bash
python scripts/data_overview.py
python src/eda/analyze.py
python src/automl/train.py
```

## 旧版项目结构（已过时）

```text
perovskite-eda/
├── data/
│   ├── raw/
│   │   └── perovskite_database_all.csv
│   ├── processed/
│   ├── data_dictionary.md
│   └── data_version.txt
├── scripts/
│   ├── data_overview.py
│   └── download_data.sh
├── src/
│   ├── eda/
│   │   └── analyze.py
│   └── automl/
│       └── train.py
├── reports/
│   └── eda_report.md
├── docs/
│   └── METHODOLOGY.md
├── tests/
├── requirements.txt
└── README.md
```

## 当前建议

请返回主 README，按 `Makefile` 目标执行：

- `make research-package`
- `make research-package-pdf`
- `make research-package-verify`
- `make test-research-package`
