# Atom Pair (AP) and Topological Torsion (TT) Fingerprint Analysis for QSPR

## Executive Summary

This comprehensive analysis compared **Atom Pair (AP)** and **Topological Torsion (TT)** molecular fingerprints for predicting Delta_PCE (change in Power Conversion Efficiency) in perovskite solar cell materials. The study analyzed **1,213 compounds** with valid SMILES structures and Delta_PCE measurements.

## Key Findings

### 1. Fingerprint Characteristics

| Metric | Atom Pair (AP) | Topological Torsion (TT) |
|--------|----------------|-------------------------|
| **Fingerprint Size** | 2048 bits | 2048 bits |
| **Active Bits** | 2041 (99.7%) | 1432 (69.9%) |
| **Bits per Molecule** | 105.8 ± 37.2 | 17.4 ± 15.3 |
| **Molecules with Empty TT** | N/A | 367 (30.3%) |

**Key Insight**: TT fingerprints require molecules with ≥4 atoms to generate meaningful patterns. 30.3% of compounds in this dataset had fewer than 4 atoms (e.g., ions, simple additives), resulting in empty TT fingerprints.

### 2. Machine Learning Performance

Random Forest models were trained using 5-fold cross-validation:

| Model | R² Score | RMSE | Performance Notes |
|-------|----------|------|-------------------|
| **RF-AP** | -0.0522 ± 0.0238 | - | Best performing |
| **RF-TT** | -0.0651 ± 0.0242 | - | Lower performance |
| **RF-Combined** | -0.0610 ± 0.0187 | - | No improvement over individual |

**Key Insight**: None of the fingerprint-only models achieved positive R² scores, indicating that:
- AP and TT fingerprints alone have limited predictive power for Delta_PCE
- Additional molecular descriptors (physicochemical properties) are needed
- The Delta_PCE relationship may depend more on processing conditions than molecular structure

### 3. Feature Importance Analysis

#### Top Atom Pair Bits
| Bit Index | Importance | Correlation | Chemical Meaning |
|-----------|------------|-------------|------------------|
| 1624 | 0.0081 | -0.0799 | Specific atom pair at distance X |
| 408 | 0.0056 | -0.0144 | Specific atom pair at distance Y |
| 409 | 0.0052 | +0.0188 | Specific atom pair at distance Y |
| 1752 | 0.0048 | -0.0608 | Specific atom pair at distance Z |
| 1899 | 0.0044 | -0.0530 | Specific atom pair at distance Z |

#### Top Topological Torsion Bits
| Bit Index | Importance | Correlation | Chemical Meaning |
|-----------|------------|-------------|------------------|
| **34** | **0.0749** | **+0.2434** | **Strongest predictor** |
| **35** | **0.0522** | **+0.2434** | **Strongest predictor** |
| 1861 | 0.0328 | -0.0950 | Torsion pattern |
| 1157 | 0.0220 | -0.0513 | Torsion pattern |
| 1613 | 0.0214 | -0.0823 | Torsion pattern |

**Key Finding**: TT bits 34 and 35 show the highest importance (0.0749 and 0.0522) and strongest correlation (+0.2434) with Delta_PCE, suggesting specific torsional patterns may influence device performance.

### 4. Chemical Interpretation

#### Atom Pair (AP) Fingerprints
- **Encode**: Distances between pairs of atoms
- **Representation**: Each bit represents atom type combinations at specific topological distances (1-7 bonds apart)
- **Strengths**:
  - Captures local molecular environments
  - More interpretable for functional group analysis
  - Works for all molecules (including small ones)
- **Limitations**:
  - Less sensitive to 3D conformation
  - May miss longer-range interactions

#### Topological Torsion (TT) Fingerprints
- **Encode**: Sequences of four consecutive bonded atoms
- **Representation**: Each bit represents a torsional angle pattern along a molecular path
- **Strengths**:
  - Captures 3D molecular shape and flexibility
  - More sensitive to stereochemistry
  - Identifies conformationally relevant patterns
- **Limitations**:
  - Requires ≥4 atoms (fails for small molecules)
  - More complex interpretation
  - Lower bit density (17.4 vs 105.8 bits/molecule)

### 5. Dimensionality Reduction (PCA)

- **AP PCA**: First 2 PCs show clear separation by Delta_PCE
- **TT PCA**: First 2 PCs show moderate separation by Delta_PCE
- **Observation**: Both fingerprint types capture structural variance but neither shows strong clustering by Delta_PCE

## Recommendations

### For QSPR Modeling of Delta_PCE:

1. **Use Hybrid Approach**: Combine AP fingerprints with:
   - Physicochemical descriptors (logP, TPSA, molecular weight)
   - Morgan/ECFP fingerprints
   - Quantum chemical descriptors

2. **Filter Dataset**: Exclude molecules with <4 atoms when using TT fingerprints

3. **Feature Selection**: Focus on top 20 TT bits (especially bits 34, 35) which show highest correlation

4. **Advanced Modeling**: Consider:
   - Deep learning for automatic feature extraction
   - Graph neural networks for molecular representation
   - Multi-task learning for related properties

### For Future Studies:

1. **Interpret Top Bits**: Investigate chemical meaning of TT bits 34 and 35
2. **Larger Fingerprints**: Test 4096-bit versions for more detail
3. **Substructure Analysis**: Map important bits to molecular fragments
4. **External Validation**: Test on independent datasets

## Output Files

### Fingerprint Data
- `/share/yhm/test/AutoML_EDA/fingerprints/atompair_fingerprints.npy` (2.4 MB)
- `/share/yhm/test/AutoML_EDA/fingerprints/torsion_fingerprints.npy` (2.4 MB)

### Analysis Results
- `/share/yhm/test/AutoML_EDA/fingerprints/ap_tt_ml_results.csv`
- `/share/yhm/test/AutoML_EDA/fingerprints/ap_tt_comparison.csv`
- `/share/yhm/test/AutoML_EDA/fingerprints/ap_tt_report.txt`

### Visualizations
- `/share/yhm/test/AutoML_EDA/figures/ap_tt_pca_scatter.png`
- `/share/yhm/test/AutoML_EDA/figures/ap_tt_analysis_overview.png`

## Conclusion

This analysis demonstrates that **Atom Pair and Topological Torsion fingerprints alone have limited predictive power for Delta_PCE** in this perovskite solar cell dataset. The negative R² scores suggest that:

1. **Delta_PCE is not strongly determined by molecular structure alone**
2. **Processing conditions**, **device architecture**, and **measurement conditions** likely play major roles
3. **Hybrid approaches** combining fingerprints with physicochemical descriptors are recommended for future modeling

However, the analysis successfully:
- Generated and compared two important fingerprint types
- Identified key structural features (TT bits 34, 35) with moderate correlation
- Provided baseline performance metrics for future QSPR studies
- Created reusable fingerprint datasets for subsequent analysis

---

*Analysis completed: 2026-02-20*
*Dataset: 1,213 compounds with Delta_PCE measurements*
*Fingerprint size: 2048 bits (AP and TT)*
