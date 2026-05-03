#!/usr/bin/env python3
"""
explore_data_sources.py
Layer 1 — Data Source Exploration for PSC additive/modulator data.

Loads the merged chemical + JV dataset, computes Delta_PCE target,
performs PSC-specific data cleaning, and runs full profiling via
reusable data_profiler module.

Runnable:
    python explore_data_sources.py

Outputs are saved to the same directory as this script.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

# Ensure data_profiler is importable from same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from data_profiler import (
    missing_value_summary,
    numeric_summary,
    correlation_matrix,
    outlier_summary,
    plot_distributions,
    plot_correlation_heatmap,
    plot_missing_matrix,
    profile_dataframe,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = "/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx"
OUTPUT_DIR = SCRIPT_DIR

CHEMICAL_COLS = [
    "cas_number", "pubchem_id", "smiles", "molecular_formula",
    "molecular_weight", "h_bond_donors", "h_bond_acceptors",
    "rotatable_bonds", "tpsa", "log_p"
]

JV_COLS = [
    "jv_reverse_scan_pce_without_modulator",
    "jv_reverse_scan_pce",
    "jv_reverse_scan_j_sc",
    "jv_reverse_scan_v_oc",
    "jv_reverse_scan_ff",
    "jv_hysteresis_index",
]

TARGET_COL = "Delta_PCE"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "exploration.log"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _xlsx_to_csv(xlsx_path: str, csv_path: str) -> None:
    """Convert .xlsx to .csv using openpyxl read-only mode for speed."""
    logger.info("Converting %s -> %s (one-time, may take ~3 min)...", xlsx_path, csv_path)
    import csv as csv_mod
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            writer.writerow(row)
            if i % 10000 == 0 and i > 0:
                logger.info("  Converted %d rows...", i)
    wb.close()
    logger.info("Conversion complete: %s", csv_path)


def load_data(path: str, nrows: Optional[int] = None,
              cache_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load Excel or CSV/Parquet with robust fallback.
    For large .xlsx files, automatically caches as CSV for fast subsequent reads.
    """
    if not os.path.exists(path):
        logger.error("Data file not found: %s", path)
        return None

    if cache_dir is None:
        cache_dir = os.path.join(SCRIPT_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    ext = os.path.splitext(path)[1].lower()
    try:
        # Fast path: CSV/Parquet direct read
        if ext == ".csv":
            df = pd.read_csv(path, nrows=nrows, low_memory=False)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext in (".xlsx", ".xls"):
            # If nrows is requested, read directly (no caching needed)
            if nrows is not None:
                df = pd.read_excel(path, nrows=nrows, engine="openpyxl")
            else:
                # Use CSV cache for full-file reads
                cache_name = os.path.splitext(os.path.basename(path))[0] + ".csv"
                cache_path = os.path.join(cache_dir, cache_name)
                if not os.path.exists(cache_path):
                    _xlsx_to_csv(path, cache_path)
                df = pd.read_csv(cache_path, low_memory=False)
        else:
            df = pd.read_excel(path, nrows=nrows, engine="openpyxl")
        logger.info("Loaded %d rows × %d cols from %s", len(df), len(df.columns), path)
        return df
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
        return None


def generate_synthetic_data(n_rows: int = 5354, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mimics the PSC merged data structure.
    Used as fallback when the real file is unavailable.
    """
    logger.warning("Generating synthetic data (n=%d) as fallback.", n_rows)
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_rows):
        mol_weight = rng.uniform(50, 800)
        hbd = int(rng.poisson(2))
        hba = int(rng.poisson(3))
        rot_bonds = int(rng.poisson(4))
        tpsa = rng.uniform(0, 200)
        logp = rng.normal(2, 1.5)

        pce_wo = rng.beta(2, 5) * 25  # 0-25 %
        delta = rng.normal(0.8, 1.5)
        pce = pce_wo + delta
        jsc = rng.beta(2, 3) * 30
        voc = rng.beta(2, 2) * 1.3
        ff = rng.beta(3, 2)
        hi = rng.exponential(0.05)

        rows.append({
            "cas_number": f"{rng.randint(10000,999999)}-{rng.randint(10,99)}-{rng.randint(0,9)}",
            "pubchem_id": rng.randint(1, 150_000_000),
            "smiles": "C" * rng.randint(2, 20),  # dummy SMILES
            "molecular_formula": f"C{rng.randint(2,20)}H{rng.randint(2,40)}",
            "molecular_weight": round(mol_weight, 2),
            "h_bond_donors": hbd,
            "h_bond_acceptors": hba,
            "rotatable_bonds": rot_bonds,
            "tpsa": round(tpsa, 2),
            "log_p": round(logp, 3),
            "jv_reverse_scan_pce_without_modulator": round(pce_wo, 3),
            "jv_reverse_scan_pce": round(pce, 3),
            "jv_reverse_scan_j_sc": round(jsc, 3),
            "jv_reverse_scan_v_oc": round(voc, 3),
            "jv_reverse_scan_ff": round(ff, 3),
            "jv_hysteresis_index": round(hi, 4),
        })
    df = pd.DataFrame(rows)
    df[TARGET_COL] = df["jv_reverse_scan_pce"] - df["jv_reverse_scan_pce_without_modulator"]
    return df


# ---------------------------------------------------------------------------
# Agent-discovered optimal cleaning strategy (VeryLoose)
# ---------------------------------------------------------------------------
# Agent exploration found that for Delta_PCE prediction, sample size is the
# dominant factor. Strict filtering reduces sample size too much and harms
# model performance (R² drops to negative). The optimal strategy is:
#   - Non-empty SMILES only (no RDKit strict validation needed)
#   - Physical PCE bounds [0, 30]
#   - All observed Delta_PCE [-5, 15]
#   - No JV consistency check
#   - No device info filter
#   - No literature quality filter
#   - No deduplication
#   - No descriptor bounds
# Result: ~5,034 rows retained, Delta_PCE R² = +0.28 (with baseline feat)
# ---------------------------------------------------------------------------

AGENTIC_VERYLOOSE_STRATEGY = {
    "smiles_validity": "NonEmpty",
    "baseline_pce_bounds": "Physical_Only",
    "treated_pce_bounds": "Physical_Only",
    "delta_pce_bounds": "AllObserved",
    "jv_consistency": "PCE_Only",
    "device_info_completeness": "NoDeviceFilter",
    "literature_quality": "NoLiteratureFilter",
    "deduplication": "NoDeduplication",
    "descriptor_bounds": "NoFilter",
}


def _filter_numeric_range(df: pd.DataFrame, col: str, bounds):
    """Filter dataframe to rows where col is within bounds."""
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[(df[col] >= bounds[0]) & (df[col] <= bounds[1])].copy()


def clean_psc_data_agentic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agent-optimal cleaning: maximize sample size while enforcing physical bounds.
    Discovered via agentic exploration of 20+ strategy combinations.
    """
    logger.info("Starting AGENTIC cleaning (VeryLoose strategy) on %d rows.", len(df))
    before = len(df)
    dfc = df.copy()

    # 1. SMILES: non-empty only
    dfc = dfc[dfc["smiles"].notna() & (dfc["smiles"].astype(str).str.len() > 3)].copy()
    logger.info("After SMILES non-empty filter: %d -> %d", before, len(dfc))
    before = len(dfc)

    # 2. Baseline PCE: physical bounds [0, 30]
    dfc = _filter_numeric_range(dfc, "jv_reverse_scan_pce_without_modulator", (0.0, 30.0))
    logger.info("After baseline PCE [0,30]: %d -> %d", before, len(dfc))
    before = len(dfc)

    # 3. Treated PCE: physical bounds [0, 30]
    dfc = _filter_numeric_range(dfc, "jv_reverse_scan_pce", (0.0, 30.0))
    logger.info("After treated PCE [0,30]: %d -> %d", before, len(dfc))
    before = len(dfc)

    # 4. Delta PCE: all observed [-5, 15]
    dfc["delta_pce"] = (pd.to_numeric(dfc["jv_reverse_scan_pce"], errors="coerce") -
                        pd.to_numeric(dfc["jv_reverse_scan_pce_without_modulator"], errors="coerce"))
    dfc = _filter_numeric_range(dfc, "delta_pce", (-5.0, 15.0))
    logger.info("After delta PCE [-5,15]: %d -> %d", before, len(dfc))

    # 5. Coerce chemical descriptors to numeric (strip units)
    for col in ["molecular_weight", "log_p", "tpsa", "h_bond_donors", "h_bond_acceptors", "rotatable_bonds"]:
        if col in dfc.columns and not pd.api.types.is_numeric_dtype(dfc[col]):
            cleaned = dfc[col].astype(str).str.replace(r"\s*g/mol\s*$", "", regex=True)
            cleaned = cleaned.str.replace(r"\s*Da\s*$", "", regex=True)
            cleaned = cleaned.str.replace(r"\s*Å²\s*$", "", regex=True)
            dfc[col] = pd.to_numeric(cleaned, errors="coerce")

    logger.info("Finished agentic cleaning: %d -> %d rows (%.2f%% retained).",
                len(df), len(dfc), 100 * len(dfc) / len(df) if len(df) > 0 else 0)
    return dfc


# ---------------------------------------------------------------------------
# PSC-specific cleaning (traditional)
# ---------------------------------------------------------------------------
def clean_psc_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-aware cleaning strategies for PSC device data.
    
    NOTE: Agentic exploration found that overly strict traditional cleaning
    reduces sample size too much, harming Delta_PCE prediction R².
    Use clean_psc_data_agentic() for the agent-optimal strategy.
    """
    logger.info("Starting PSC data cleaning on %d rows.", len(df))
    before = len(df)

    # Make a copy to avoid SettingWithCopy
    dfc = df.copy()

    # Coerce known numeric columns (CSV re-import may leave them as object)
    numeric_cols_to_coerce = [
        "jv_reverse_scan_pce_without_modulator", "jv_reverse_scan_pce",
        "jv_reverse_scan_j_sc", "jv_reverse_scan_v_oc", "jv_reverse_scan_ff",
        "jv_hysteresis_index", "molecular_weight", "h_bond_donors",
        "h_bond_acceptors", "rotatable_bonds", "tpsa", "log_p"
    ]
    for col in numeric_cols_to_coerce:
        if col in dfc.columns and not pd.api.types.is_numeric_dtype(dfc[col]):
            logger.info("Coercing column '%s' to numeric (was %s).", col, dfc[col].dtype)
            dfc[col] = pd.to_numeric(dfc[col], errors="coerce")

    # 1. Ensure target-related columns exist
    pce_wo_col = "jv_reverse_scan_pce_without_modulator"
    pce_col = "jv_reverse_scan_pce"

    if pce_wo_col in dfc.columns and pce_col in dfc.columns:
        dfc = dfc.dropna(subset=[pce_wo_col, pce_col], how="all")
        logger.info("After dropping rows with both PCE missing: %d -> %d", before, len(dfc))
        before = len(dfc)

    # 2. Clamp PCE
    for col in [pce_wo_col, pce_col]:
        if col in dfc.columns:
            invalid = ((dfc[col] < 0) | (dfc[col] > 30)).sum()
            if invalid > 0:
                logger.info("Clamping %d out-of-range values in %s", invalid, col)
                dfc[col] = dfc[col].clip(lower=0, upper=30)

    # 3. Non-positive Jsc / Voc
    for col in ["jv_reverse_scan_j_sc", "jv_reverse_scan_v_oc"]:
        if col in dfc.columns:
            bad = (dfc[col] <= 0).sum()
            if bad > 0:
                logger.info("Removing %d rows with %s <= 0", bad, col)
                dfc = dfc[dfc[col] > 0]

    # 4. FF bounds
    ff_col = "jv_reverse_scan_ff"
    if ff_col in dfc.columns:
        bad = ((dfc[ff_col] < 0) | (dfc[ff_col] > 1)).sum()
        if bad > 0:
            logger.info("Clamping %d out-of-range FF values", bad)
            dfc[ff_col] = dfc[ff_col].clip(lower=0, upper=1)

    # 5. Extreme hysteresis
    hi_col = "jv_hysteresis_index"
    if hi_col in dfc.columns:
        extreme = (dfc[hi_col] > 0.5).sum()
        if extreme > 0:
            logger.info("Flagging %d rows with hysteresis_index > 0.5 (kept but flagged)", extreme)
            dfc["hysteresis_extreme_flag"] = (dfc[hi_col] > 0.5).astype(int)
        else:
            dfc["hysteresis_extreme_flag"] = 0

    # 6. Deduplicate on chemical + baseline PCE keys
    dedup_keys = [k for k in ["cas_number", "smiles", pce_wo_col] if k in dfc.columns]
    if dedup_keys:
        dups = dfc.duplicated(subset=dedup_keys).sum()
        if dups > 0:
            logger.info("Dropping %d duplicate rows on %s", dups, dedup_keys)
            dfc = dfc.drop_duplicates(subset=dedup_keys)

    # 7. molecular_weight to numeric
    if "molecular_weight" in dfc.columns:
        dfc["molecular_weight"] = pd.to_numeric(dfc["molecular_weight"], errors="coerce")

    # 8. Drop missing SMILES if present and many are missing
    if "smiles" in dfc.columns:
        missing_smiles = dfc["smiles"].isnull().sum()
        if missing_smiles > 0:
            logger.info("Dropping %d rows with missing SMILES", missing_smiles)
            dfc = dfc.dropna(subset=["smiles"])

    logger.info("Finished cleaning: %d -> %d rows.", before, len(dfc))
    return dfc


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add PSC-relevant derived columns."""
    dfc = df.copy()
    pce_wo = "jv_reverse_scan_pce_without_modulator"
    pce = "jv_reverse_scan_pce"
    if pce_wo in dfc.columns and pce in dfc.columns:
        dfc[TARGET_COL] = dfc[pce] - dfc[pce_wo]
        logger.info("Computed target '%s' = %s - %s", TARGET_COL, pce, pce_wo)
    # Relative PCE improvement
    if pce_wo in dfc.columns and pce in dfc.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            dfc["rel_pce_improvement"] = (dfc[pce] - dfc[pce_wo]) / dfc[pce_wo].replace(0, np.nan)
        dfc["rel_pce_improvement"] = dfc["rel_pce_improvement"].replace([np.inf, -np.inf], np.nan)
    return dfc


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PSC Data Source Exploration")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH,
                        help="Path to merged data file")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force use of synthetic data generator")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit rows read from file (for quick tests)")
    parser.add_argument("--out-dir", type=str, default=OUTPUT_DIR,
                        help="Directory to write outputs")
    parser.add_argument("--cleaning-strategy", type=str, default="agentic",
                        choices=["agentic", "traditional"],
                        help="Data cleaning strategy: 'agentic' (default, optimal for Delta_PCE) or 'traditional' (stricter)")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    logger.info("=== PSC Data Source Exploration started at %s ===", datetime.now().isoformat())
    logger.info("Output directory: %s", out_dir)
    logger.info("Cleaning strategy: %s", args.cleaning_strategy)

    # Load or synthesize
    if args.synthetic:
        df = generate_synthetic_data()
    else:
        df = load_data(args.data_path, nrows=args.nrows)
        if df is None:
            logger.warning("Real data unavailable; falling back to synthetic.")
            df = generate_synthetic_data()

    # ------------------------------------------------------------------
    # Phase 1: Raw profiling
    # ------------------------------------------------------------------
    logger.info("Phase 1: Profiling raw data (%d rows, %d cols).", len(df), len(df.columns))
    raw_profile_dir = os.path.join(out_dir, "raw_profile")
    os.makedirs(raw_profile_dir, exist_ok=True)
    raw_artifacts = profile_dataframe(df, name="raw", out_dir=raw_profile_dir)
    logger.info("Raw profile artifacts: %s", raw_artifacts)

    # ------------------------------------------------------------------
    # Phase 2: Cleaning + derived features
    # ------------------------------------------------------------------
    logger.info("Phase 2: Cleaning and deriving features (strategy=%s).", args.cleaning_strategy)
    if args.cleaning_strategy == "agentic":
        df_clean = clean_psc_data_agentic(df)
    else:
        df_clean = clean_psc_data(df)
    df_clean = add_derived_features(df_clean)

    # Save cleaned data
    cleaned_path = os.path.join(out_dir, "cleaned_data.csv")
    df_clean.to_csv(cleaned_path, index=False)
    logger.info("Saved cleaned data to %s (%d rows).", cleaned_path, len(df_clean))

    # ------------------------------------------------------------------
    # Phase 3: Cleaned profiling
    # ------------------------------------------------------------------
    logger.info("Phase 3: Profiling cleaned data.")
    clean_profile_dir = os.path.join(out_dir, "clean_profile")
    os.makedirs(clean_profile_dir, exist_ok=True)
    clean_artifacts = profile_dataframe(df_clean, name="clean", out_dir=clean_profile_dir)
    logger.info("Clean profile artifacts: %s", clean_artifacts)

    # ------------------------------------------------------------------
    # Phase 4: PSC-specific summaries
    # ------------------------------------------------------------------
    logger.info("Phase 4: PSC-specific summaries.")

    # Chemical columns summary
    chem_present = [c for c in CHEMICAL_COLS if c in df_clean.columns]
    if chem_present:
        chem_numeric = [c for c in chem_present if pd.api.types.is_numeric_dtype(df_clean[c])]
        chem_summary = numeric_summary(df_clean, cols=chem_numeric)
        chem_path = os.path.join(out_dir, "chemical_descriptor_summary.csv")
        if not chem_summary.empty:
            chem_summary.to_csv(chem_path)
            logger.info("Saved chemical descriptor summary to %s", chem_path)

    # JV columns summary
    jv_present = [c for c in JV_COLS if c in df_clean.columns]
    if jv_present:
        jv_summary = numeric_summary(df_clean, cols=jv_present)
        jv_path = os.path.join(out_dir, "jv_metrics_summary.csv")
        if not jv_summary.empty:
            jv_summary.to_csv(jv_path)
            logger.info("Saved JV metrics summary to %s", jv_path)

    # Target summary
    if TARGET_COL in df_clean.columns:
        target_summary = numeric_summary(df_clean, cols=[TARGET_COL])
        target_path = os.path.join(out_dir, "target_delta_pce_summary.csv")
        if not target_summary.empty:
            target_summary.to_csv(target_path)
            logger.info("Saved target '%s' summary to %s", TARGET_COL, target_path)

    # Correlation among key numeric columns
    key_cols = [c for c in (CHEMICAL_COLS + JV_COLS + [TARGET_COL]) if c in df_clean.columns]
    key_numeric = [c for c in key_cols if pd.api.types.is_numeric_dtype(df_clean[c])]
    if len(key_numeric) >= 2:
        corr = correlation_matrix(df_clean, cols=key_numeric)
        corr_path = os.path.join(out_dir, "key_features_correlation.csv")
        corr.to_csv(corr_path)
        plot_correlation_heatmap(corr, os.path.join(out_dir, "key_features_correlation_heatmap.png"))
        logger.info("Saved key features correlation to %s", corr_path)

    # Outlier summary on key numeric columns
    outlier_df = outlier_summary(df_clean, cols=key_numeric)
    outlier_path = os.path.join(out_dir, "key_features_outlier_summary.csv")
    if not outlier_df.empty:
        outlier_df.to_csv(outlier_path, index=False)
        logger.info("Saved outlier summary to %s", outlier_path)

    # ------------------------------------------------------------------
    # Phase 5: Final report JSON
    # ------------------------------------------------------------------
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_path": args.data_path,
        "synthetic_used": args.synthetic or not os.path.exists(args.data_path),
        "cleaning_strategy": args.cleaning_strategy,
        "raw_shape": {"rows": int(len(df)), "cols": int(len(df.columns))},
        "clean_shape": {"rows": int(len(df_clean)), "cols": int(len(df_clean.columns))},
        "target_col": TARGET_COL,
        "cleaned_data_path": cleaned_path,
        "artifacts": {
            "raw_profile": raw_artifacts,
            "clean_profile": clean_artifacts,
        },
    }
    report_path = os.path.join(out_dir, "exploration_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved exploration report to %s", report_path)
    logger.info("=== Exploration finished at %s ===", datetime.now().isoformat())


if __name__ == "__main__":
    main()
