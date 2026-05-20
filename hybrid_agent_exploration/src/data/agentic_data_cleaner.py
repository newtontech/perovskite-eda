#!/usr/bin/env python3
"""
agentic_data_cleaner.py

Agent驱动的数据清洗管线 —— 让Agent自主探索最优筛选策略。

设计理念：
  1. 不给Agent固定规则，而是给Agent"操作空间"和"评估方法"
  2. Agent尝试不同的筛选策略组合
  3. 每种策略组合后，评估下游ML模型性能（5-Fold CV R²）
  4. Agent记录实验结果，比较效果，调整策略
  5. Agent迭代进化，找到使R²最大化的筛选组合

Agent Loop:
  观察数据 → 选择筛选动作 → 执行清洗 → 评估效果（ML R²）
  → 记录实验 → 反思 → 调整策略 → 下一轮
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "cleaning_action_space.yaml"
RAW_DATA_PATH = PROJECT_ROOT / "explorations" / "data_source_exploration" / ".cache" / "merged_llm_crossref_data_streaming_with_chemical_data_fast.csv"
LOG_DIR = PROJECT_ROOT / "results" / "exploration_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CleaningExperiment:
    """Record of one cleaning strategy experiment."""
    experiment_id: int
    timestamp: str
    strategy: Dict[str, str]          # e.g., {"smiles_validity": "strict", "baseline_pce_bounds": "medium", ...}
    n_raw: int
    n_cleaned: int
    retention_rate: float
    delta_pce_r2: float               # 5-fold CV R² predicting Delta_PCE (no baseline feat)
    delta_pce_r2_with_baseline: float # 5-fold CV R² predicting Delta_PCE (with baseline feat)
    abs_pce_r2: float                 # 5-fold CV R² predicting absolute PCE
    rmse_delta: float
    mae_delta: float
    notes: str = ""


@dataclass
class CleanerState:
    """Agent's internal state (memory)."""
    experiment_history: List[CleaningExperiment] = field(default_factory=list)
    best_strategy: Optional[Dict[str, str]] = None
    best_r2: float = -float("inf")
    iteration: int = 0


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw data from cached CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[Agent] Loaded raw data: {len(df)} rows × {len(df.columns)} cols")
    return df


# ---------------------------------------------------------------------------
# 2. Atomic Cleaning Operations
# ---------------------------------------------------------------------------

_SMILES_CACHE = {}

def _mol_from_smiles_cached(s: str):
    if s in _SMILES_CACHE:
        return _SMILES_CACHE[s]
    mol = Chem.MolFromSmiles(s)
    _SMILES_CACHE[s] = mol
    return mol

def filter_smiles_validity(df: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "NoFilter":
        return df
    if level == "NonEmpty":
        return df[df["smiles"].notna() & (df["smiles"].astype(str).str.len() > 3)].copy()
    # strict: RDKit valid — batch process with cache
    raw_smiles = df["smiles"]
    # Fast pre-filter: drop NaN-like and very short strings
    fast_mask = raw_smiles.notna()
    smiles_str = raw_smiles.astype(str)
    fast_mask &= (smiles_str.str.len() > 3)
    fast_mask &= ~smiles_str.isin(["nan", "None", "null", "NA", ""])
    candidates_idx = df[fast_mask].index
    if len(candidates_idx) == 0:
        return df.iloc[0:0].copy()
    # Batch RDKit validation with cache
    valid_mask = []
    for s in df.loc[candidates_idx, "smiles"].astype(str):
        valid_mask.append(_mol_from_smiles_cached(s) is not None)
    valid_idx = candidates_idx[valid_mask]
    return df.loc[valid_idx].copy()


def filter_numeric_range(df: pd.DataFrame, col: str, bounds: Tuple[float, float]) -> pd.DataFrame:
    # Clean common unit suffixes before numeric conversion
    if df[col].dtype == object:
        cleaned = df[col].astype(str).str.replace(r"\s*g/mol\s*$", "", regex=True)
        cleaned = cleaned.str.replace(r"\s*Da\s*$", "", regex=True)
        cleaned = cleaned.str.replace(r"\s*Å²\s*$", "", regex=True)
        cleaned = cleaned.str.replace(r"\s*deg C\s*$", "", regex=True)
        cleaned = cleaned.str.replace(r"\s*%\s*$", "", regex=True)
        df[col] = pd.to_numeric(cleaned, errors="coerce")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[(df[col] >= bounds[0]) & (df[col] <= bounds[1])].copy()


def _get_level_cfg(action_space: Dict, action_name: str, level: str) -> Dict:
    """Look up level config by key or by 'name' field."""
    levels = action_space["actions"][action_name]["levels"]
    if level in levels:
        return levels[level]
    for key, cfg in levels.items():
        if isinstance(cfg, dict) and cfg.get("name") == level:
            return cfg
    raise KeyError(f"Level '{level}' not found in {action_name}")


def filter_baseline_pce(df: pd.DataFrame, level: str, action_space: Dict) -> pd.DataFrame:
    if level == "Physical_Only":
        return filter_numeric_range(df, "jv_reverse_scan_pce_without_modulator", (0.0, 30.0))
    bounds = _get_level_cfg(action_space, "baseline_pce_bounds", level)["bounds"]
    return filter_numeric_range(df, "jv_reverse_scan_pce_without_modulator", tuple(bounds))


def filter_treated_pce(df: pd.DataFrame, level: str, action_space: Dict) -> pd.DataFrame:
    if level == "Physical_Only":
        return filter_numeric_range(df, "jv_reverse_scan_pce", (0.0, 30.0))
    bounds = _get_level_cfg(action_space, "treated_pce_bounds", level)["bounds"]
    return filter_numeric_range(df, "jv_reverse_scan_pce", tuple(bounds))


def filter_delta_pce(df: pd.DataFrame, level: str, action_space: Dict) -> pd.DataFrame:
    # Always compute delta_pce first
    df["delta_pce"] = pd.to_numeric(df["jv_reverse_scan_pce"], errors="coerce") - pd.to_numeric(df["jv_reverse_scan_pce_without_modulator"], errors="coerce")
    if level == "AllObserved":
        return filter_numeric_range(df, "delta_pce", (-5.0, 15.0))
    bounds = _get_level_cfg(action_space, "delta_pce_bounds", level)["bounds"]
    return df[(df["delta_pce"] >= bounds[0]) & (df["delta_pce"] <= bounds[1])].copy()


def filter_jv_consistency(df: pd.DataFrame, level: str, action_space: Dict) -> pd.DataFrame:
    if level == "PCE_Only":
        return df
    cfg = _get_level_cfg(action_space, "jv_consistency", level)
    for col_suffix in ["_without_modulator", ""]:
        for param, bounds_key in [("jv_reverse_scan_j_sc", "jsc_bounds"),
                                   ("jv_reverse_scan_v_oc", "voc_bounds"),
                                   ("jv_reverse_scan_ff", "ff_bounds")]:
            col = param + col_suffix
            if col in df.columns:
                df = filter_numeric_range(df, col, tuple(cfg[bounds_key]))
    return df


def filter_descriptors(df: pd.DataFrame, level: str, action_space: Dict) -> pd.DataFrame:
    if level == "NoFilter":
        return df
    cfg = _get_level_cfg(action_space, "descriptor_bounds", level)
    for col, bounds in cfg.items():
        if col == "name" or col == "rule" or col == "rationale":
            continue
        if col in df.columns:
            df = filter_numeric_range(df, col, tuple(bounds))
    return df


def filter_device_info(df: pd.DataFrame, level: str, action_space: Dict) -> pd.DataFrame:
    if level == "NoDeviceFilter":
        return df
    cfg = _get_level_cfg(action_space, "device_info_completeness", level)
    for col in cfg["required_cols"]:
        if col in df.columns:
            df = df[df[col].notna()]
            for ex in cfg.get("exclude_values", []):
                df = df[df[col].astype(str) != ex]
    return df.copy()


def filter_deduplication(df: pd.DataFrame, level: str, action_space: Dict) -> pd.DataFrame:
    if level == "NoDeduplication":
        return df
    cfg = _get_level_cfg(action_space, "deduplication", level)
    subset = [c for c in cfg["subset"] if c in df.columns]
    if subset:
        return df.drop_duplicates(subset=subset).copy()
    return df.copy()


def filter_literature_quality(df: pd.DataFrame, level: str, action_space: Dict) -> pd.DataFrame:
    if level == "NoLiteratureFilter":
        return df
    cfg = _get_level_cfg(action_space, "literature_quality", level)
    for f in cfg.get("filters", []):
        if "journal is not null" in f and "journal" in df.columns:
            df = df[df["journal"].notna()]
        if "doi is not null" in f and "doi" in df.columns:
            df = df[df["doi"].notna()]
        if "title is not null" in f and "title" in df.columns:
            df = df[df["title"].notna()]
        if "publication_date is not null" in f and "publication_date" in df.columns:
            df = df[df["publication_date"].notna()]
    return df.copy()


def compute_rdkit_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RDKit descriptors and add to dataframe (uses cache)."""
    rows = []
    for s in df["smiles"].astype(str):
        mol = _mol_from_smiles_cached(s)
        if mol is None:
            rows.append({"MolWt": np.nan, "LogP": np.nan, "TPSA": np.nan,
                         "HBD": np.nan, "HBA": np.nan, "RotBonds": np.nan})
            continue
        try:
            rows.append({
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "RotBonds": Descriptors.NumRotatableBonds(mol),
            })
        except Exception:
            rows.append({"MolWt": np.nan, "LogP": np.nan, "TPSA": np.nan,
                         "HBD": np.nan, "HBA": np.nan, "RotBonds": np.nan})
    desc_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# 3. Strategy Execution
# ---------------------------------------------------------------------------

def execute_strategy(df_raw: pd.DataFrame, strategy: Dict[str, str], action_space: Dict) -> pd.DataFrame:
    """Execute a cleaning strategy on raw data."""
    df = df_raw.copy()
    
    # Step 1: SMILES validity
    df = filter_smiles_validity(df, strategy.get("smiles_validity", "RDKit_Strict"))
    
    # Step 2: Baseline PCE
    df = filter_baseline_pce(df, strategy.get("baseline_pce_bounds", "Standard_Devices"), action_space)
    
    # Step 3: Treated PCE
    df = filter_treated_pce(df, strategy.get("treated_pce_bounds", "Standard_Treated"), action_space)
    
    # Step 4: Delta PCE
    df = filter_delta_pce(df, strategy.get("delta_pce_bounds", "ReasonableRange"), action_space)
    
    # Step 5: JV consistency
    df = filter_jv_consistency(df, strategy.get("jv_consistency", "FullJV_Standard"), action_space)
    
    # Step 6: Device info
    df = filter_device_info(df, strategy.get("device_info_completeness", "StructureAndComposition"), action_space)
    
    # Step 7: Literature quality
    df = filter_literature_quality(df, strategy.get("literature_quality", "HasPublicationInfo"), action_space)
    
    # Step 8: Deduplication
    df = filter_deduplication(df, strategy.get("deduplication", "Unique_Molecule_Baseline"), action_space)
    
    # Step 9: Compute RDKit descriptors
    df = compute_rdkit_descriptors(df)
    
    # Step 10: Descriptor bounds (applied after computation)
    df = filter_descriptors(df, strategy.get("descriptor_bounds", "DrugLike_Standard"), action_space)
    
    return df


# ---------------------------------------------------------------------------
# 4. Evaluation: Downstream ML Performance
# ---------------------------------------------------------------------------

def evaluate_ml(df: pd.DataFrame, feature_cols: List[str], target_col: str, 
                n_splits: int = 5, random_state: int = 42) -> Dict[str, float]:
    """Evaluate RF 5-fold CV on given features/target."""
    if target_col not in df.columns or len(df) < n_splits * 2:
        return {"r2": -999, "rmse": 999, "mae": 999}
    
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    if len(df) < n_splits * 2:
        return {"r2": -999, "rmse": 999, "mae": 999}
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_r2, fold_rmse, fold_mae = [], [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_val)
        fold_r2.append(r2_score(y_val, y_pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        fold_mae.append(mean_absolute_error(y_val, y_pred))
    
    return {
        "r2": np.mean(fold_r2),
        "rmse": np.mean(fold_rmse),
        "mae": np.mean(fold_mae),
    }


def evaluate_strategy(df_clean: pd.DataFrame) -> Dict[str, float]:
    """Evaluate a cleaned dataset on three target schemes."""
    desc_cols = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]
    baseline_col = "jv_reverse_scan_pce_without_modulator"
    
    # Ensure delta_pce exists
    if "delta_pce" not in df_clean.columns:
        df_clean["delta_pce"] = (pd.to_numeric(df_clean["jv_reverse_scan_pce"], errors="coerce") - 
                                  pd.to_numeric(df_clean["jv_reverse_scan_pce_without_modulator"], errors="coerce"))
    
    # Scheme 1: Delta_PCE, no baseline feature
    res1 = evaluate_ml(df_clean, desc_cols, "delta_pce")
    
    # Scheme 2: Delta_PCE, with baseline feature
    res2 = evaluate_ml(df_clean, desc_cols + [baseline_col], "delta_pce")
    
    # Scheme 3: Absolute PCE, with baseline feature
    res3 = evaluate_ml(df_clean, desc_cols + [baseline_col], "jv_reverse_scan_pce")
    
    return {
        "delta_pce_r2": res1["r2"],
        "delta_pce_r2_with_baseline": res2["r2"],
        "abs_pce_r2": res3["r2"],
        "rmse_delta": res1["rmse"],
        "mae_delta": res1["mae"],
    }


# ---------------------------------------------------------------------------
# 5. Agent Strategy Generators
# ---------------------------------------------------------------------------

def generate_predefined_strategies() -> List[Dict[str, str]]:
    """Generate a set of predefined strategies ranging from strict to loose."""
    return [
        # Strategy 0: Ultra-Strict (literature gold standard)
        {
            "name": "UltraStrict_Literature",
            "smiles_validity": "RDKit_Strict",
            "baseline_pce_bounds": "HighQuality_Devices",
            "treated_pce_bounds": "HighQuality_Treated",
            "delta_pce_bounds": "EffectiveOnly",
            "jv_consistency": "FullJV_Strict",
            "device_info_completeness": "FullDeviceInfo",
            "literature_quality": "PeerReviewed_Only",
            "deduplication": "Unique_Device",
            "descriptor_bounds": "DrugLike_Strict",
        },
        # Strategy 1: Strict
        {
            "name": "Strict",
            "smiles_validity": "RDKit_Strict",
            "baseline_pce_bounds": "HighQuality_Devices",
            "treated_pce_bounds": "HighQuality_Treated",
            "delta_pce_bounds": "ReasonableRange",
            "jv_consistency": "FullJV_Strict",
            "device_info_completeness": "FullDeviceInfo",
            "literature_quality": "HasPublicationInfo",
            "deduplication": "Unique_Device",
            "descriptor_bounds": "DrugLike_Strict",
        },
        # Strategy 2: Standard (balanced)
        {
            "name": "Standard",
            "smiles_validity": "RDKit_Strict",
            "baseline_pce_bounds": "Standard_Devices",
            "treated_pce_bounds": "Standard_Treated",
            "delta_pce_bounds": "ReasonableRange",
            "jv_consistency": "FullJV_Standard",
            "device_info_completeness": "StructureAndComposition",
            "literature_quality": "HasPublicationInfo",
            "deduplication": "Unique_Molecule_Baseline",
            "descriptor_bounds": "DrugLike_Standard",
        },
        # Strategy 3: Loose
        {
            "name": "Loose",
            "smiles_validity": "RDKit_Strict",
            "baseline_pce_bounds": "AllValid_Devices",
            "treated_pce_bounds": "AllValid_Treated",
            "delta_pce_bounds": "WideRange",
            "jv_consistency": "PCE_Only",
            "device_info_completeness": "NoDeviceFilter",
            "literature_quality": "NoLiteratureFilter",
            "deduplication": "Unique_SMILES",
            "descriptor_bounds": "DrugLike_Standard",
        },
        # Strategy 4: Very Loose
        {
            "name": "VeryLoose",
            "smiles_validity": "NonEmpty",
            "baseline_pce_bounds": "Physical_Only",
            "treated_pce_bounds": "Physical_Only",
            "delta_pce_bounds": "AllObserved",
            "jv_consistency": "PCE_Only",
            "device_info_completeness": "NoDeviceFilter",
            "literature_quality": "NoLiteratureFilter",
            "deduplication": "NoDeduplication",
            "descriptor_bounds": "NoFilter",
        },
        # Strategy 5: Delta-PCE Optimized (focus on positive gains)
        {
            "name": "DeltaOptimized",
            "smiles_validity": "RDKit_Strict",
            "baseline_pce_bounds": "Standard_Devices",
            "treated_pce_bounds": "Standard_Treated",
            "delta_pce_bounds": "EffectiveOnly",
            "jv_consistency": "FullJV_Standard",
            "device_info_completeness": "StructureAndComposition",
            "literature_quality": "HasPublicationInfo",
            "deduplication": "Unique_Molecule_Baseline",
            "descriptor_bounds": "DrugLike_Standard",
        },
        # Strategy 6: High-Baseline Focus (mature devices only)
        {
            "name": "HighBaselineFocus",
            "smiles_validity": "RDKit_Strict",
            "baseline_pce_bounds": "HighQuality_Devices",
            "treated_pce_bounds": "Standard_Treated",
            "delta_pce_bounds": "ReasonableRange",
            "jv_consistency": "FullJV_Standard",
            "device_info_completeness": "StructureAndComposition",
            "literature_quality": "HasPublicationInfo",
            "deduplication": "Unique_Molecule_Baseline",
            "descriptor_bounds": "DrugLike_Standard",
        },
    ]


def agent_self_reflect(history: List[CleaningExperiment]) -> Dict[str, str]:
    """
    Agent self-reflection: analyze history and propose next strategy.
    This is a simple rule-based reflection; in a full LLM-agent system,
    this would be replaced by an LLM reasoning step.
    """
    if not history:
        return generate_predefined_strategies()[2]  # Start with Standard
    
    # Find best so far
    best = max(history, key=lambda x: x.delta_pce_r2_with_baseline)
    
    # Simple heuristics for next strategy
    if best.retention_rate < 0.05:
        # Too strict, loosen up
        return generate_predefined_strategies()[2]  # Standard
    elif best.retention_rate > 0.50:
        # Too loose, tighten up
        return generate_predefined_strategies()[1]  # Strict
    else:
        # Good balance, try a variant
        return generate_predefined_strategies()[5]  # DeltaOptimized


# ---------------------------------------------------------------------------
# 6. Main Agent Loop
# ---------------------------------------------------------------------------

def run_agentic_cleaning(n_experiments: int = 7):
    """Run the agentic cleaning pipeline."""
    print("=" * 70)
    print("AGENTIC DATA CLEANING PIPELINE")
    print("=" * 70)
    print()
    print("Philosophy: The Agent explores different cleaning strategies,")
    print("evaluates each by downstream ML performance, and evolves.")
    print()
    
    # Load action space
    with open(CONFIG_PATH) as f:
        action_space = yaml.safe_load(f)
    
    # Load raw data
    df_raw = load_raw_data()
    
    state = CleanerState()
    strategies = generate_predefined_strategies()
    
    print(f"[Agent] Will run {min(n_experiments, len(strategies))} experiments")
    print()
    
    for i in range(min(n_experiments, len(strategies))):
        strategy = strategies[i]
        strategy_name = strategy.pop("name")
        
        print(f"-" * 70)
        print(f"Experiment {i+1}: Strategy = '{strategy_name}'")
        print(f"-" * 70)
        print(f"  Strategy config: {json.dumps(strategy, indent=2)}")
        
        # Execute cleaning
        t0 = time.time()
        df_clean = execute_strategy(df_raw, strategy, action_space)
        clean_time = time.time() - t0
        
        n_raw = len(df_raw)
        n_clean = len(df_clean)
        retention = n_clean / n_raw if n_raw > 0 else 0
        
        print(f"  Cleaning: {n_raw} → {n_clean} rows ({retention*100:.2f}% retained, {clean_time:.1f}s)")
        
        if n_clean < 50:
            print(f"  ⚠️ Too few samples ({n_clean}), skipping ML evaluation")
            metrics = {"delta_pce_r2": -999, "delta_pce_r2_with_baseline": -999, 
                      "abs_pce_r2": -999, "rmse_delta": 999, "mae_delta": 999}
        else:
            # Evaluate downstream ML
            t1 = time.time()
            metrics = evaluate_strategy(df_clean)
            eval_time = time.time() - t1
            print(f"  ML eval: {eval_time:.1f}s")
            print(f"  Delta_PCE R² (no baseline feat):  {metrics['delta_pce_r2']:+.4f}")
            print(f"  Delta_PCE R² (with baseline feat): {metrics['delta_pce_r2_with_baseline']:+.4f}")
            print(f"  Absolute PCE R²:                  {metrics['abs_pce_r2']:+.4f}")
            print(f"  RMSE (Delta): {metrics['rmse_delta']:.4f}")
        
        # Record experiment
        exp = CleaningExperiment(
            experiment_id=i+1,
            timestamp=pd.Timestamp.now().isoformat(),
            strategy={**strategy, "name": strategy_name},
            n_raw=n_raw,
            n_cleaned=n_clean,
            retention_rate=retention,
            delta_pce_r2=metrics["delta_pce_r2"],
            delta_pce_r2_with_baseline=metrics["delta_pce_r2_with_baseline"],
            abs_pce_r2=metrics["abs_pce_r2"],
            rmse_delta=metrics["rmse_delta"],
            mae_delta=metrics["mae_delta"],
            notes="",
        )
        state.experiment_history.append(exp)
        
        # Update best
        if metrics["delta_pce_r2_with_baseline"] > state.best_r2:
            state.best_r2 = metrics["delta_pce_r2_with_baseline"]
            state.best_strategy = {**strategy, "name": strategy_name}
            print(f"  ★ NEW BEST! Delta_PCE R² = {state.best_r2:+.4f}")
        
        print()
    
    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    summary = pd.DataFrame([
        {
            "Exp": e.experiment_id,
            "Strategy": e.strategy.get("name", "unknown"),
            "N_Cleaned": e.n_cleaned,
            "Retention%": f"{e.retention_rate*100:.1f}",
            "Delta_R2": f"{e.delta_pce_r2:+.4f}",
            "Delta_R2+Baseline": f"{e.delta_pce_r2_with_baseline:+.4f}",
            "AbsPCE_R2": f"{e.abs_pce_r2:+.4f}",
            "RMSE": f"{e.rmse_delta:.3f}",
        }
        for e in state.experiment_history
    ])
    print(summary.to_string(index=False))
    
    print()
    print(f"BEST STRATEGY: '{state.best_strategy.get('name')}'")
    print(f"  Delta_PCE R² (with baseline): {state.best_r2:+.4f}")
    print(f"  Config: {json.dumps(state.best_strategy, indent=2)}")
    print()
    
    # Save best cleaned data
    best_df = execute_strategy(df_raw, {k: v for k, v in state.best_strategy.items() if k != "name"}, action_space)
    best_path = PROJECT_ROOT / "explorations" / "data_source_exploration" / "agentic_best_cleaned.csv"
    best_df.to_csv(best_path, index=False)
    print(f"Saved best cleaned data: {best_path} ({len(best_df)} rows)")
    
    # Save experiment log
    log_path = LOG_DIR / "agentic_cleaning_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "best_strategy": state.best_strategy,
            "best_r2": state.best_r2,
            "experiments": [asdict(e) for e in state.experiment_history],
        }, f, indent=2, default=str)
    print(f"Saved experiment log: {log_path}")
    
    return state, best_df


if __name__ == "__main__":
    run_agentic_cleaning()
