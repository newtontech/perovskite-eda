# Perovskite Solar Cells - Exploratory Data Analysis (EDA)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Exploratory Data Analysis for Perovskite Solar Cells Database**

This project performs comprehensive Exploratory Data Analysis (EDA) and AutoML machine learning analysis on the perovskite solar cells database.

---

## Data Source

- **Database**: `Perovskite_database_content_all_data.csv` (81 MB)
- **Source**: [Jesperkemist/perovskitedatabase_data](https://github.com/Jesperkemist/perovskitedatabase_data)
- **Records**: 41,447 samples
- **Features**: 410 columns

---

## Quick Start

### Install Dependencies

```bash
uv sync
```

### Run Analysis

```bash
# Data cleaning
uv run python scripts/clean_data.py

# EDA analysis
uv run python scripts/eda_analysis.py

# AutoML analysis
uv run python automl/analyze_with_automl.py
```

---

## Key Results

### Performance Statistics

| Metric | Value |
|--------|-------|
| Max PCE | 36.20% |
| Mean PCE | 12.05% |
| Dominant Material | MA-Pb (68.5%) |
| Dominant Architecture | nip (62.8%) |

### AutoML Model Performance

| Model | R2 | RMSE | MAE |
|-------|-----|------|-----|
| **Random Forest** | **0.9687** | **0.8676** | **0.3851** |
| LightGBM | 0.9681 | 0.8758 | 0.4006 |
| XGBoost | 0.9666 | 0.8963 | 0.4461 |
| Gradient Boosting | 0.9639 | 0.9317 | 0.4972 |
| Linear Regression | 0.8501 | 1.8984 | 1.2828 |

### Feature Importance

| Feature | Importance |
|---------|------------|
| JV_default_Jsc | 63.94% |
| JV_default_FF | 26.52% |
| JV_default_Voc | 8.54% |
| Perovskite_thickness | 0.46% |
| Perovskite_band_gap | 0.35% |

---

## Project Structure

```
perovskite-eda/
├── data/
│   ├── raw/                    # Raw data
│   │   └── Perovskite_database_content_all_data.csv
│   └── processed/              # Processed data
│       ├── perovskite_cleaned.csv
│       ├── perovskite_ml_ready.csv
│       └── data_report.md
├── scripts/
│   ├── clean_data.py           # Data cleaning
│   └── eda_analysis.py         # EDA analysis
├── automl/
│   └── analyze_with_automl.py  # AutoML analysis
├── figures/
│   ├── temporal/               # Temporal trends
│   ├── materials/              # Material analysis
│   ├── performance/            # Performance analysis
│   ├── correlations/           # Correlation matrix
│   └── automl/                 # AutoML results
├── reports/
│   ├── eda_report.md           # EDA report
│   └── automl/
│       ├── automl_report.md    # AutoML report
│       └── feature_importance.csv
├── pyproject.toml
└── README.md
```

---

## Analysis Contents

### 1. Temporal Analysis
- Publication year distribution (2009-2025)
- PCE efficiency evolution over time

### 2. Material Analysis
- Perovskite type distribution (MA-Pb, MAFA-Pb, FA-Pb, etc.)
- Cell architecture distribution (nip, pin, etc.)
- Dimension types (3D, 2D, 2D/3D mixed)

### 3. Performance Metrics
- PCE (Power Conversion Efficiency)
- Voc (Open Circuit Voltage)
- Jsc (Short Circuit Current)
- FF (Fill Factor)

### 4. AutoML Machine Learning
- 6 model comparison (Random Forest, XGBoost, LightGBM, etc.)
- Best model R2 = 96.87%
- Feature importance analysis

---

## Tech Stack

- **Python 3.11+**
- **Pandas** - Data processing
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Machine learning
- **XGBoost/LightGBM** - Gradient boosting models

---

## Reports

- [EDA Report](reports/eda_report.md)
- [AutoML Report](reports/automl/automl_report.md)

---

## Related Projects

- [Perovskite_Database_Multiagents](https://github.com/newtontech/Perovskite_Database_Multiagents)
- [perovskite_literature_rag](https://github.com/newtontech/perovskite_literature_rag)
- [Perovskite_Pretrain_Models](https://github.com/newtontech/Perovskite_Pretrain_Models)

---

## License

MIT License

---

**Last Updated**: 2026-03-12
**Version**: 2.1.0 (Integrated perovskitedatabase_data)
