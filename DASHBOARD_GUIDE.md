# QSPR Analysis Dashboard - User Guide

## Overview

This interactive HTML dashboard provides a comprehensive visualization of the Quantitative Structure-Property Relationship (QSPR) analysis results for perovskite solar cell modulators.

**Dashboard Location:** `/share/yhm/test/AutoML_EDA/dashboard.html`

## Key Statistics Summary

| Metric | Value |
|--------|-------|
| **Total Samples** | 5,354 modulator measurements |
| **Unique Compounds** | 1,035 chemical entities |
| **Delta_PCE Mean** | 1.893% efficiency improvement |
| **Delta_PCE Range** | [-19.190%, +44.800%] |
| **Positive Results** | 82% showing improvement |
| **Best Predictor** | Log P (r = 0.083, p < 0.001) |
| **Best ML Model** | Random Forest (ROC-AUC: 0.624) |

## Dashboard Sections

### 1. Overview (Key Statistics)
- **Purpose:** Quick summary of dataset and main findings
- **Content:** 6 key metrics displayed in stat cards
- **Highlight:** Total samples, Delta_PCE mean, positive results percentage

### 2. Key Findings
- **Purpose:** Executive summary of major discoveries
- **Content:**
  - Lipophilicity as primary predictor
  - Molecular flexibility importance
  - Non-linear relationship insights
- **Interactive Tabs:** Top Features, Key Substructures, Design Guidelines

### 3. Data Statistics
- **Purpose:** Descriptive statistics for all molecular descriptors
- **Content:** Mean, median, std dev, min, max, skewness
- **Key Insight:** All descriptors show significant right-skewness

### 4. Correlation Analysis
- **Purpose:** Statistical correlations with Delta_PCE
- **Content:** Pearson correlation with 95% confidence intervals
- **Significant Features:**
  1. Log P: r = 0.083 (p < 0.001)
  2. Rotatable Bonds: r = 0.063 (p < 0.001)
  3. Molecular Weight: r = 0.056 (p < 0.001)
  4. H-Bond Acceptors: r = 0.053 (p < 0.001)

### 5. Distribution Analysis
- **Purpose:** Assess normality and distribution characteristics
- **Content:** Normality tests, distribution plots
- **Key Finding:** Delta_PCE is NOT normally distributed (p < 2.12×10⁻⁵⁵)

### 6. Outlier Detection
- **Purpose:** Identify anomalous data points
- **Methods:** IQR, Z-score, Isolation Forest
- **Results:** 9.97% multivariate outliers detected

### 7. Quartile Analysis
- **Purpose:** Examine descriptor-performance relationships
- **Content:** ANOVA results across Delta_PCE quartiles
- **Key Finding:** Log P shows strongest association (F = 9.70, p < 0.001)

### 8. Fingerprint Analysis
- **Purpose:** Substructure-level analysis
- **Content:** MACCS keys correlation and RF importance
- **Top Substructures:**
  - Aromatic H on heteroatom (+0.147)
  - Thiophene ring (+0.094)
  - 1,2-diol motif (+0.084)

### 9. ML Model Results
- **Purpose:** Machine learning performance comparison
- **Content:** ROC-AUC, accuracy, F1-score
- **Best Model:** Random Forest with MACCS keys

### 10. Chemical Space Analysis
- **Purpose:** Visualize chemical diversity
- **Content:** PCA 2D/3D plots, loadings, pair plots
- **Variance Explained:** PC1 = 53.63%, PC2 = 22.07%

### 11. Additional Visualizations
- **Purpose:** Comprehensive figure gallery
- **Content:** Distribution plots, correlation plots, fingerprint visualizations
- **Organization:** Tabbed interface by category

### 12. Download Data
- **Purpose:** Export analysis results
- **Available Files:**
  - Processed dataset (CSV)
  - Correlation statistics
  - Distribution statistics
  - ANOVA results
  - ML model results
  - Analysis reports (MD)

## Design Guidelines from Analysis

Based on the statistical findings, optimal modulators should have:

1. **Moderate Lipophilicity:** Log P between 0.5 and 3.0
2. **Controlled Flexibility:** 2-5 rotatable bonds
3. **Aromatic Heterocycles:** Thiophene, pyridine, or similar π-systems
4. **Hydrogen Bonding Capacity:** Diol or similar motifs
5. **Intermediate Molecular Weight:** 150-300 Da

Features to avoid:
- Excessive polycyclic aromatic systems
- Unusual elements (B, Si, P) without clear rationale
- Extremely high or low Log P values

## Technical Details

### Dashboard Technology
- **Framework:** Pure HTML/CSS/JavaScript
- **Charts:** Plotly.js (interactive)
- **Responsive Design:** Mobile and desktop compatible
- **Browser Support:** All modern browsers

### File Size
- **Dashboard HTML:** ~80 KB
- **Dependencies:** Plotly.js loaded from CDN
- **Images:** Linked from `/share/yhm/test/AutoML_EDA/figures/`

### Color Scheme
- **Primary:** #2563eb (Blue)
- **Secondary:** #8b5cf6 (Purple)
- **Success:** #10b981 (Green)
- **Warning:** #f59e0b (Amber)
- **Danger:** #ef4444 (Red)
- **Background:** Dark theme (#0f172a)

## Interactive Features

### Navigation
- Fixed sidebar with smooth scrolling
- Auto-highlighting of current section
- Quick access to all sections

### Tabs
- Multiple tabbed interfaces for organizing content
- Instant switching between related views
- Visual feedback for active tabs

### Charts
- Interactive Plotly.js charts
- Hover for detailed values
- Zoom and pan capabilities
- Responsive resizing

### Images
- Click to view in modal
- Full-resolution display
- Keyboard (Escape) to close

### Downloads
- Direct download links for all data files
- Organized by category
- File size and format indicated

## Usage Instructions

### Viewing the Dashboard

1. **Local Access:**
   ```bash
   # Using Python's built-in server
   cd /share/yhm/test/AutoML_EDA
   python -m http.server 8000

   # Open in browser
   # http://localhost:8000/dashboard.html
   ```

2. **Direct File Opening:**
   - Open `dashboard.html` directly in any modern web browser
   - Note: Some browsers may restrict local file access

### Navigation Tips
- Use sidebar for quick section navigation
- Click any tab to switch views
- Click figures to view full-size
- Use download buttons to export data

### Printing/Export
- Use browser's print function (Ctrl+P / Cmd+P)
- Select "Save as PDF" for offline viewing
- Dark theme may require print style adjustment

## Data Sources

### Primary Data
- **Dataset:** `/share/yhm/test/AutoML_EDA/processed_data.csv`
- **Samples:** 5,354 measurements
- **Features:** 6 molecular descriptors + Delta_PCE

### Figures
- **Location:** `/share/yhm/test/AutoML_EDA/figures/`
- **Total:** 30+ visualization files
- **Formats:** PNG (high-resolution)

### Tables
- **Location:** `/share/yhm/test/AutoML_EDA/tables/`
- **Files:** 7 CSV files with statistics

### Reports
- **Main Report:** `/share/yhm/test/AutoML_EDA/report/main_report.md`
- **Supporting Info:** `/share/yhm/test/AutoML_EDA/report/supporting_information.md`
- **Summaries:** Various `.md` files in root directory

## Statistical Methods Used

### Correlation Analysis
- Pearson correlation (linear relationships)
- Spearman rank correlation (monotonic)
- Kendall's tau (ordinal)
- Bonferroni correction for multiple comparisons

### Distribution Analysis
- Shapiro-Wilk normality test
- D'Agostino-Pearson test
- Skewness and kurtosis calculation
- Q-Q plots

### Outlier Detection
- Interquartile range (IQR) method
- Z-score method
- Isolation Forest (multivariate)

### Quartile Analysis
- Delta_PCE stratified into quartiles
- One-way ANOVA for descriptor differences
- F-statistic testing

### Machine Learning
- Random Forest classification
- Logistic Regression with L1 regularization
- 5-fold cross-validation
- ROC-AUC evaluation

### Fingerprint Analysis
- MACCS keys (166 structural bits)
- Point-biserial correlation
- Feature importance ranking

## Key Scientific Findings

### Primary Predictors
1. **Log P (Lipophilicity)**
   - Strongest statistical correlation
   - Mechanism: Interface compatibility, moisture resistance
   - Optimal range: 0.5 - 3.0

2. **Rotatable Bonds (Flexibility)**
   - Significant positive correlation
   - Mechanism: Conformational adaptability
   - Optimal: 2-5 bonds

3. **Molecular Weight (Size)**
   - Significant but weak correlation
   - Mechanism: Surface coverage
   - Optimal: 150-300 Da

4. **H-Bond Acceptors**
   - Significant weak correlation
   - Mechanism: Defect passivation
   - Electron-donating capacity

### Beneficial Substructures
- Aromatic heterocycles (thiophene, pyridine)
- Diol motifs (1,2-diol, 1,3-diol)
- Nitro groups
- Disulfide bridges

### Detrimental Features
- Multiple aromatic rings
- Excessive polycyclic systems
- Unusual elements without clear rationale

## Limitations and Considerations

### Statistical Limitations
- All significant correlations are weak (r < 0.1)
- Linear methods explain limited variance
- Multicollinearity among descriptors

### Data Limitations
- Only 6 basic descriptors examined
- Context dependence not captured
- Experimental confounders present

### Model Limitations
- Negative R² for some fingerprint models
- Overfitting concerns with high-dimensional features
- External validation needed

## Future Directions

### Recommended Analyses
1. **Non-linear Methods:** Neural networks, GNNs
2. **Quantum Descriptors:** HOMO/LUMO, dipole moment
3. **3D Descriptors:** Molecular volume, surface area
4. **Scaffold Validation:** Robust model assessment

### Experimental Validation
1. Prioritize high Log P modulators
2. Test flexibility hypothesis
3. Investigate substructure effects
4. Explore synergistic combinations

## Citation Information

When using this dashboard or its data, please cite:

```
QSPR Analysis of Perovskite Solar Cell Modulators (2026).
Analysis performed using ML-Master framework.
Dataset: 5,354 modulator measurements.
DOI: [To be assigned]
```

## Support and Contact

For questions about the dashboard or analysis:

- **Framework:** ML-Master (https://github.com/sjtu-sai-agents/ML-Master)
- **Analysis Date:** February 20, 2026
- **Location:** `/share/yhm/test/AutoML_EDA/`

## Version History

- **v1.0** (2026-02-20): Initial dashboard release
  - Complete statistical analysis
  - Interactive visualizations
  - Download capabilities
  - Responsive design

---

**Note:** This dashboard was generated automatically as part of the QSPR analysis pipeline. All statistical interpretations should be verified by domain experts before drawing scientific conclusions.
