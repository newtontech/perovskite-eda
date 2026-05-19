# Gap Analysis: Current Report vs JPCL Reference

## Executive Summary

| Metric | JPCL Reference | Current Report | Gap | Priority |
|--------|---------------|----------------|-----|----------|
| Main text figures | 8 (Fig 1-8) | 5 (Fig 1-5) | -3 figures | High |
| SI tables | 5 (S1-S5) | 0 structured tables | Complete absence | High |
| SI figures | 13 (S1-S13) | 25 (mostly repeats) | Missing per-model SHAP + validation structures | Medium |
| References | 37 | 0 | Complete absence | Critical |
| Model configs | 12 (3 algo × 4 features) | 4 (2 algo × 2 features) | -8 configs | High |
| External validation | 10 SAMs with <10% error | None | Missing | High |
| Molecular design | 3 designed molecules | None | Missing | High |
| Data cleaning audit | Table S1 (9 removed) | "No data removed" | Missing audit | Medium |
| Feature documentation | Table S3 (26 features) | Generic descriptions | Missing detailed docs | Medium |
| Hyperparameter table | Table S4 | Default params only | Missing tuned params | Medium |
| CV details | Table S5 (10-fold × 12 models) | "not available" | Missing | Medium |

---

## Main Text Structure Comparison

| Section | JPCL | Current | Gap | Action Required |
|---------|------|---------|-----|-----------------|
| **Abstract** | ~150 words, includes: SAM motivation, ML method, key metrics (R²=0.506), SHAP top 5, external validation, 3 designed molecules | ~80 words, basic metrics (R²=0.299), mentions SHAP but no specifics | Missing: SAM context, specific top features, external validation, molecular design | Expand to ≥150 words |
| **Introduction** | 4 paragraphs: (1) PSC efficiency progress, (2) SAM advantages, (3) ML challenges in materials, (4) This work scope | 1 paragraph: generic ML + PSC background | Missing: SAM-specific motivation, literature milestones (Science 2021, Nature 2022, Joule 2024), small dataset challenge discussion | Add 3 paragraphs with citations |
| **Methods** | Detailed: data source (91 SAMs), cleaning (9 removed), features (RDKit 196 + Morgan 256/512/1024), models (RF/XGB/GBDT), hyperparameter grid, 10-fold CV, SHAP, external validation set, molecular design strategy | 4 subsections: brief mention of cleaning, features, models, SHAP | Missing: sample size, cleaning details, hyperparameter search space, external validation protocol, molecular design rules | Expand to match JPCL depth |
| **Results** | 4 subsections: (2.1) 12-model performance + feature comparison + algorithm comparison, (2.2) SHAP analysis (bar + dependence + interaction), (2.3) External validation, (2.4) Molecular design | 3 subsections: optimal config, feature impact, interpretability (generic) | Missing: 12-model grid, per-algorithm comparison, external validation, molecular design | Add subsections |
| **Discussion** | Integrated into Results (2.1-2.4) | Separate "Limitations" paragraph | Missing: scaffold-split discussion, comparison with Co-PAS | Expand discussion |
| **Conclusion** | Summary of findings + implications | Generic conclusion | Missing: specific designed molecules, future directions | Add molecular design + external validation plans |

---

## Figure Comparison

| JPCL Figure | Content | Current Equivalent | Gap |
|-------------|---------|-------------------|-----|
| **Fig 1** | Workflow diagram (data → features → models → SHAP → design) | None | Missing workflow figure |
| **Fig 2** | 12-model parity grid (3×4): RDKit_XGB, RDKit_RF, RDKit_GBDT, Morgan256_XGB, Morgan512_XGB, Morgan1024_XGB, Morgan256_RF, Morgan512_RF, Morgan1024_RF, Morgan256_GBDT, Morgan512_GBDT, Morgan1024_GBDT | matrix_fig01_prediction_grid (4 models) | Need 12-model grid |
| **Fig 3** | Performance bars: RMSE/R²/r/MAE for all 12 models | matrix_fig02_performance_comparison (4 models) | Need 12-model bars |
| **Fig 4** | SHAP summary bar + top 10 features | fig_interpretability (2×2 SHAP composite) | Similar, but need top 10 feature names |
| **Fig 5** | Single feature dependence (4 features, 2×2) | Included in fig_interpretability (c) | Similar |
| **Fig 6** | Dual feature interaction (4 pairs, 2×2) | Included in fig_interpretability (d) | Similar |
| **Fig 7** | External validation scatter + bar chart | None | Missing entirely |
| **Fig 8** | 3 designed SAM molecules + predicted PCE | None | Missing entirely |

---

## SI Comparison

### Tables

| JPCL SI Table | Content | Current Equivalent | Gap |
|---------------|---------|-------------------|-----|
| **S1** | 9 removed low-PCE data points (name, SMILES, structure, PCE) | "No data points were removed" | Need actual cleaning audit |
| **S2** | Complete SAM dataset (name, SMILES, structure, PCE) | Generic description | Need dataset table |
| **S3** | 26 features with meanings | Generic feature descriptions | Need per-feature documentation |
| **S4** | Hyperparameter tuning results per model | Default parameters only | Need GridSearch/Optuna results |
| **S5** | 10-fold CV metrics per model | "not available" | Need CV fold details |

### Figures

| JPCL SI Figure | Content | Current Equivalent | Gap |
|----------------|---------|-------------------|-----|
| **S1** | Data distribution histogram | si_fig21_data_distribution | ✅ Similar |
| **S2-S12** | Per-model SHAP top 10 (11 models) | si_shap_bar_0, si_shap_swarm_0 (2 models) | Need 10 more per-model SHAP |
| **S13** | SAM molecular structures for validation | None | Need molecular structure depictions |

---

## References Comparison

### JPCL References: 37 papers

**Coverage by subfield:**
- PSC fundamentals / SAM reviews: (1)-(13) — 13 papers
- ML methods / materials informatics: (14)-(28) — 15 papers
- SHAP / interpretability: (29)-(31) — 3 papers
- Molecular design / screening: (32)-(33) — 2 papers
- Experimental validation / recent SAMs: (34)-(37) — 4 papers

**Key missing topics in current report:**
1. SAM/HTM design literature (Science 2021, Nature 2022, Joule 2024)
2. ML for PSC review papers (Liu & Yan series)
3. Co-PAS / virtual screening (Pu et al. 2025)
4. Self-driving lab / HTE (Cakan et al. 2024)
5. Perovskite Database Project (Jacobsson et al. 2022)
6. Scaffold / temporal split evaluation
7. Uni-Mol / JTVAE learned representations
8. Active learning / Bayesian optimization

---

## Critical Actions Required

### Phase 1: Reference Infrastructure (Highest Priority)
- [ ] Install and verify `scholar-gateway` / `perovskite_rag` for literature retrieval
- [ ] Retrieve ≥30 references covering 5+ subfields
- [ ] Build citation database with DOI validation
- [ ] Integrate references into main text (Introduction, Methods, Results, Discussion)

### Phase 2: Experimental Matrix Expansion (High Priority)
- [ ] Run `jpcl_sam_matrix` (12 configs: 3 algo × 4 features)
- [ ] Generate 12-model parity grid (Fig 2 equivalent)
- [ ] Generate 12-model performance bars (Fig 3 equivalent)
- [ ] Generate per-model SHAP for SI (S2-S12 equivalent)

### Phase 3: Missing Analysis Modules (High Priority)
- [ ] External validation: identify 5-10 recent molecules from literature
- [ ] Virtual screening: predict on candidate library, rank top-k
- [ ] Molecular design: scaffold modification based on SHAP insights
- [ ] Generate external validation figure (Fig 7 equivalent)
- [ ] Generate designed molecules figure (Fig 8 equivalent)

### Phase 4: SI Tables (Medium Priority)
- [ ] Table S1: Data cleaning audit
- [ ] Table S2: Complete molecular dataset
- [ ] Table S3: Feature descriptions with physical meanings
- [ ] Table S4: Hyperparameter tuning results
- [ ] Table S5: Cross-validation fold details

### Phase 5: Harness Engineering (Medium Priority)
- [ ] Sandbox: timeout + memory limits
- [ ] Observability: structured JSONL logging
- [ ] Retry: transient failure recovery
- [ ] Guardrail: input/output validation + reference deduplication
