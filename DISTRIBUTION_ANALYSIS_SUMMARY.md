# Comprehensive Distribution Analysis for QSPR Data

## Executive Summary

This report presents a comprehensive distribution analysis of the QSPR (Quantitative Structure-Property Relationship) dataset for perovskite solar cell additives. The analysis examines the distribution characteristics of Delta_PCE (the primary target variable) and six key molecular descriptors, providing insights crucial for subsequent modeling and interpretation.

**Analysis Date:** February 20, 2026
**Dataset Size:** 5,354 samples
**Analysis Framework:** Python (scipy, sklearn, matplotlib, seaborn)

---

## 1. Delta_PCE Distribution Analysis

### Basic Statistics
- **Mean:** 1.893%
- **Median:** 1.600%
- **Standard Deviation:** 3.460%
- **Range:** -19.190% to 44.800%
- **Skewness:** 1.0115 (moderately right-skewed)
- **Kurtosis:** 11.9092 (heavy-tailed, leptokurtic)

### Normality Assessment
Both Shapiro-Wilk and D'Agostino-Pearson tests conclusively reject the null hypothesis of normality:

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Shapiro-Wilk | W = 0.8572 | p < 2.12×10⁻⁵⁵ | **NOT Normal** |
| D'Agostino-Pearson | χ² = 1681.69 | p < 1×10⁻³⁰⁰ | **NOT Normal** |

**Key Findings:**
- Delta_PCE exhibits a **non-normal distribution** with significant positive skewness
- The distribution is **leptokurtic** (heavy-tailed), indicating more extreme values than expected under normality
- Mean > Median suggests right-skewness, with a tail extending toward higher Delta_PCE values
- The Q-Q plot deviation from the diagonal line confirms non-normality

**Implications for Modeling:**
1. Consider **non-parametric methods** for statistical inference
2. **Data transformation** (e.g., log, Box-Cox) may improve normality for parametric tests
3. **Robust regression techniques** may be more appropriate than OLS
4. **Prediction intervals** should account for non-normal residuals

---

## 2. Chemical Descriptor Distributions

### Distribution Statistics Summary

| Descriptor | Mean | Median | Std | Skewness | Kurtosis | Distribution Type |
|------------|------|--------|-----|----------|----------|-------------------|
| **Molecular Weight** | 258.58 | 158.97 | 308.69 | 3.96 | 28.76 | Highly right-skewed |
| **H-Bond Donors** | 0.73 | 0.00 | 1.99 | 12.42 | 269.84 | Extremely right-skewed |
| **H-Bond Acceptors** | 1.95 | 1.00 | 3.55 | 6.68 | 96.29 | Highly right-skewed |
| **Rotatable Bonds** | 2.42 | 0.00 | 6.18 | 6.98 | 97.60 | Highly right-skewed |
| **TPSA** | 38.15 | 26.02 | 68.96 | 10.26 | 188.68 | Extremely right-skewed |
| **Log P** | 1.58 | 0.24 | 6.02 | 2.18 | 6.47 | Moderately right-skewed |

### Key Observations

1. **All descriptors exhibit significant right-skewness**
   - Skewness values range from 2.18 (Log P) to 12.42 (H-Bond Donors)
   - Median values are substantially lower than means, indicating many small values with few large outliers

2. **Heavy tails (leptokurtosis)**
   - Kurtosis values far exceed the normal distribution benchmark of 0
   - H-Bond Donors shows extreme kurtosis (269.84), suggesting many extreme values

3. **Zero-inflation observed**
   - H-Bond Donors, H-Bond Acceptors, and Rotatable Bonds have median = 0
   - Indicates many compounds lack these structural features

### Log-Transform Analysis Results

Log-transformation significantly reduces skewness for all descriptors:

| Descriptor | Original Skew | Log-Transform Skew | Improvement |
|------------|---------------|-------------------|-------------|
| Molecular Weight | 3.96 | 0.85 | 79% reduction |
| H-Bond Donors | 12.42 | 1.24 | 90% reduction |
| H-Bond Acceptors | 6.68 | 0.52 | 92% reduction |
| Rotatable Bonds | 6.98 | 1.87 | 73% reduction |
| TPSA | 10.26 | 0.71 | 93% reduction |
| Log P | 2.18 | 0.13 | 94% reduction |

**Recommendation:** Apply log-transformation to all descriptors prior to parametric modeling.

---

## 3. Outlier Detection Analysis

### Outlier Detection Results

#### IQR Method (1.5 × IQR)
| Variable | Outliers | Percentage |
|----------|----------|------------|
| Log P | 1,021 | 19.07% |
| Rotatable Bonds | 628 | 11.73% |
| Delta_PCE | 580 | 10.83% |
| Molecular Weight | 554 | 10.35% |
| H-Bond Acceptors | 465 | 8.69% |
| H-Bond Donors | 402 | 7.51% |
| TPSA | 262 | 4.89% |

#### Z-Score Method (|z| > 3)
| Variable | Outliers | Percentage |
|----------|----------|------------|
| Molecular Weight | 170 | 3.18% |
| Log P | 173 | 3.23% |
| Delta_PCE | 121 | 2.26% |
| Rotatable Bonds | 104 | 1.94% |
| H-Bond Donors | 59 | 1.10% |
| H-Bond Acceptors | 59 | 1.10% |
| TPSA | 56 | 1.05% |

#### Isolation Forest (Multivariate)
- **Total outliers detected:** 534 (9.97% of samples)
- **Contamination parameter:** 0.10
- **Method advantage:** Captures multivariate outliers not detected by univariate methods

### PCA Visualization Insights
- **PC1 explains 53.63% of variance** - primarily driven by molecular size/complexity
- **PC2 explains 22.07% of variance** - related to polarity/hydrogen bonding
- Outliers tend to cluster in specific regions of PC space, suggesting distinct chemical classes

**Recommendations:**
1. **Flag outliers** for special examination rather than automatic removal
2. **Investigate chemical nature** of multivariate outliers - may represent novel additive classes
3. **Consider robust methods** that are less sensitive to outliers
4. **Document outlier handling** strategy for reproducibility

---

## 4. Delta_PCE Quartile Analysis

### Quartile Distribution
| Quartile | Sample Size | Delta_PCE Range |
|----------|-------------|-----------------|
| Q1 (Low) | 1,341 | Negative to ~0.8% |
| Q2 | 1,343 | ~0.8% to ~1.6% |
| Q3 | 1,333 | ~1.6% to ~2.7% |
| Q4 (High) | 1,337 | ~2.7% to 44.8% |

### ANOVA Results: Descriptor Differences Across Quartiles

| Descriptor | F-Statistic | p-value | Significance | Effect Size |
|------------|-------------|---------|--------------|-------------|
| **Log P** | 9.70 | 2.20×10⁻⁶ | *** (p<0.001) | Strong |
| **Rotatable Bonds** | 4.20 | 5.61×10⁻³ | ** (p<0.01) | Moderate |
| **Molecular Weight** | 3.93 | 8.13×10⁻³ | ** (p<0.01) | Moderate |
| **H-Bond Acceptors** | 3.76 | 1.03×10⁻² | * (p<0.05) | Small |
| **H-Bond Donors** | 2.89 | 3.43×10⁻² | * (p<0.05) | Small |
| **TPSA** | 2.13 | 9.38×10⁻² | ns | Not significant |

**Key Findings:**

1. **Log P shows strongest association** with Delta_PCE performance
   - Higher Delta_PCE quartiles tend to have higher Log P values
   - Suggests lipophilicity is an important factor for additive performance

2. **Molecular flexibility matters**
   - Rotatable bonds show significant differences across quartiles
   - Optimal flexibility may enhance additive effectiveness

3. **Molecular size plays a role**
   - Molecular weight differences are statistically significant
   - However, effect size is moderate

4. **Polarity less important**
   - TPSA (Topological Polar Surface Area) shows no significant differences
   - H-bonding capacity shows only weak effects

**Implications for Additive Design:**
- Prioritize **moderately lipophilic compounds** (Log P > 0.5)
- Consider **molecular flexibility** but avoid excessive rotatable bonds
- **Molecular weight** should be optimized - not too small, not too large
- **Polar features** may be less critical than lipophilicity and flexibility

---

## 5. Statistical Recommendations for QSPR Modeling

Based on this distribution analysis, the following approaches are recommended:

### Data Preprocessing
1. **Apply log-transformation** to all molecular descriptors
2. **Consider Yeo-Johnson transformation** for Delta_PCE (handles negative values)
3. **Standardize features** after transformation (z-score normalization)
4. **Document outlier handling strategy** - flag but don't automatically remove

### Modeling Approach
1. **Use robust regression methods** (e.g., Huber, RANSAC) to handle outliers
2. **Consider non-parametric methods** (Random Forest, Gradient Boosting)
3. **Apply quantile regression** to understand predictor effects across Delta_PCE distribution
4. **Use cross-validation** with appropriate folds to account for data distribution

### Statistical Inference
1. **Avoid parametric tests** assuming normality (use Mann-Whitney, Kruskal-Wallis)
2. **Report bootstrap confidence intervals** for model coefficients
3. **Use permutation tests** for significance assessment
4. **Apply false discovery rate correction** for multiple comparisons

### Validation Strategy
1. **Stratify sampling** by Delta_PCE quartiles in train/test splits
2. **Monitor performance across Delta_PCE range** (not just overall R²)
3. **Validate on chemical diversity** (ensure test set covers chemical space)
4. **Consider external validation** on independent datasets

---

## 6. Generated Outputs

### Figures (7)
1. `distribution_delta_pce_histogram_qq.png` - Delta_PCE distribution with KDE, normal fit, and Q-Q plot
2. `distribution_descriptors_histograms.png` - Multi-panel histogram grid with KDE for all descriptors
3. `distribution_descriptors_box_violin.png` - Box plots and violin plots for descriptor comparison
4. `distribution_log_transform_analysis.png` - Original vs. log-transformed distributions for skewed features
5. `outlier_detection_summary.png` - PCA visualization with outlier detection from multiple methods
6. `distribution_quartile_comparison.png` - Box plots of descriptors by Delta_PCE quartile
7. `distribution_quartile_delta_pce.png` - Delta_PCE distribution overlay and mean comparison by quartile

### Tables (4)
1. `distribution_statistics.csv` - Complete descriptive statistics for all descriptors
2. `outlier_summary.csv` - Outlier counts by method (IQR, Z-score)
3. `quartile_statistics.csv` - Descriptor statistics by Delta_PCE quartile
4. `anova_results.csv` - ANOVA test results for quartile differences

### Report
1. `distribution_analysis_report.txt` - Complete text report of all findings

---

## 7. Conclusions

This comprehensive distribution analysis reveals several critical insights:

1. **Non-normality is pervasive** - Both target variable and all descriptors deviate significantly from normal distributions
2. **Heavy skewness requires transformation** - Log-transformation substantially improves distribution symmetry
3. **Outliers are abundant** - Up to 19% of samples show outlier characteristics depending on method
4. **Log P is most predictive** - Shows strongest statistical association with Delta_PCE performance
5. **Chemical interpretability is maintained** - Statistical findings align with chemical intuition about additive mechanisms

**Next Steps:**
1. Implement recommended data transformations
2. Build QSPR models using robust methods
3. Validate models across Delta_PCE range
4. Investigate chemical nature of outliers for novel insights

---

**Analysis performed using:** Python 3.x, scipy, sklearn, matplotlib, seaborn
**Data source:** `/share/yhm/test/AutoML_EDA/processed_data.csv`
**Analysis script:** `comprehensive_distribution_analysis.py`
