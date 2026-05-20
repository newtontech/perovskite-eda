
# JPCL Reference Paper Structure Analysis

## Paper Info
- **Title**: Machine Learning Accelerated Design of Self-Assembled Monolayers for High-Performance Perovskite Solar Cells
- **Journal**: J. Phys. Chem. Lett. 2026, 17, 3362−3373
- **Authors**: Haifeng Li, Yue Zang, Zhikang Zhu, Chenyang Zhu, Weihong Liu, Zihao Zhang, Wensheng Yan
- **Institution**: Hangzhou Dianzi University

## Main Text Structure

### Abstract (~150 words)
- Motivation: SAMs as HTMs, empirical trial-and-error is slow
- Method: ML platform combining RDKit descriptors + Morgan fingerprints
- Key results: RDKit_XGBoost optimal (RMSE=1.862, R²=0.5058, r=0.8161, MAE=1.528)
- SHAP top 5: EState_VSA5, fr_benzene, EState_VSA2, SlogP_VSA1, Chi0v
- External validation: relative errors within 10%, min 0.55%
- Designed 3 new SAM molecules with predicted PCE approaching 27%

### 1. Introduction
- Paragraph 1: PSC efficiency progress (Nature 2022 601:573, Science 2023 380:404)
- Paragraph 2: SAM advantages over conventional HTMs (Joule 2024 8:1691)
- Paragraph 3: ML in materials science, small datasets challenge
- Paragraph 4: This work — comprehensive feature space (RDKit + Morgan), 12 models, SHAP, external validation, molecular design

### 2. Results and Discussion
- **2.1 Model Performance**: 12-model matrix results
  - RDKit_XGBoost best: R²=0.5058, RMSE=1.862, r=0.8161, MAE=1.528
  - Feature comparison: RDKit > Morgan (all sizes)
  - Algorithm comparison: XGBoost > RF > GBDT
- **2.2 SHAP Analysis**:
  - Top 10 features all RDKit descriptors
  - Single feature dependence (EState_VSA5, fr_benzene, etc.)
  - Dual feature interactions
- **2.3 External Validation**:
  - 10 recently reported SAMs
  - Relative errors 0.55%−9.86%
- **2.4 Molecular Design**:
  - 3 designed molecules (MPA-MBT-BA, MPA-EBT-BA, MPA-MEBT-BA)
  - Predicted PCE up to ~27%

### 3. Methods
- Data collection: 91 SAMs from literature
- Data cleaning: removed 9 low-PCE outliers (Table S1)
- Features: RDKit descriptors (196) + Morgan fingerprints (256/512/1024)
- Models: RF, XGBoost, GBDT
- Evaluation: 10-fold CV + random split
- Hyperparameter: GridSearchCV
- SHAP: shap.Explainer
- External validation: 10 recent SAMs
- Molecular design: scaffold modification based on SHAP insights

### 4. Conclusion
- Summary of key findings
- Implications for SAM design

## Main Text Figures (8 total)
| Figure | Content | Layout |
|--------|---------|--------|
| Fig 1 | Workflow diagram | Single |
| Fig 2 | 12-model parity plots | 3×4 grid |
| Fig 3 | Performance bars (RMSE/R²/r/MAE) | 1×4 |
| Fig 4 | SHAP summary + top 10 features | Composite |
| Fig 5 | Single feature dependence (4 features) | 2×2 |
| Fig 6 | Dual feature interaction (4 pairs) | 2×2 |
| Fig 7 | External validation scatter + bar | Composite |
| Fig 8 | 3 designed molecules + predicted PCE | 1×3 |

## SI Structure

### Tables (5)
| Table | Content |
|-------|---------|
| S1 | 9 removed low-PCE data points (name, SMILES, structure, PCE) |
| S2 | SAM molecular information (name, SMILES, structure, PCE) |
| S3 | 26 input features and their meanings |
| S4 | Hyperparameter tuning results (full dataset) |
| S5 | 10-fold CV performance metrics of 12 models |

### Figures (13)
| Figure | Content |
|--------|---------|
| S1 | Data distribution histogram |
| S2-S12 | Per-model SHAP top 10 feature importance (11 models) |
| S13 | SAM molecular structures for validation |

## References (37 total)
- Format: ACS style, numbered (1)-(37)
- Coverage: PSC fundamentals, SAM/HTM design, ML methods, DFT, interface engineering
- Key journals: Science, Nature, Joule, Adv. Funct. Mater., ACS Nano, Mater. Chem. Front.
