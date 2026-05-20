#!/usr/bin/env python3
"""
agentic_evolution.py

基于第一轮Agent探索结果，从最优策略（VeryLoose）出发，
逐步收紧各个维度，寻找更优的筛选组合。

第一轮发现：
  - VeryLoose (4934 rows): Delta_PCE R² = +0.2834 (with baseline)
  - Standard (1250 rows): Delta_PCE R² = -0.0379 (with baseline)
  
结论：严格过滤导致样本量不足，R²反而下降。

本轮探索：在VeryLoose基础上，单独收紧每个维度，观察效果。
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from agentic_data_cleaner import (
    load_raw_data, execute_strategy, evaluate_strategy, CONFIG_PATH, PROJECT_ROOT, LOG_DIR
)


def run_evolution_from_baseline():
    """从VeryLoose基线出发，逐步收紧各维度。"""
    print("=" * 70)
    print("AGENTIC EVOLUTION: From VeryLoose Baseline")
    print("=" * 70)
    
    with open(CONFIG_PATH) as f:
        action_space = yaml.safe_load(f)
    
    df_raw = load_raw_data()
    
    # Base strategy (best from round 1)
    base = {
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
    
    experiments = []
    
    # Experiment 0: Baseline
    experiments.append({"name": "Baseline_VeryLoose", **base})
    
    # Experiment 1: Tighten SMILES only
    experiments.append({"name": "Tighten_SMILES", **{**base, "smiles_validity": "RDKit_Strict"}})
    
    # Experiment 2: Tighten baseline PCE to standard
    experiments.append({"name": "Tighten_Baseline_Standard", **{**base, "baseline_pce_bounds": "Standard_Devices"}})
    
    # Experiment 3: Tighten delta PCE to wide
    experiments.append({"name": "Tighten_Delta_Wide", **{**base, "delta_pce_bounds": "WideRange"}})
    
    # Experiment 4: Two-dimensional combo
    experiments.append({"name": "Combo_SMILES_Baseline_Delta", **{**base, 
        "smiles_validity": "RDKit_Strict", 
        "baseline_pce_bounds": "Standard_Devices",
        "delta_pce_bounds": "ReasonableRange"}})
    
    # Experiment 5: VeryLoose + descriptors
    experiments.append({"name": "Baseline_Plus_Descriptors", **{**base, "descriptor_bounds": "DrugLike_Standard"}})
    
    # Experiment 6: Deduplication only
    experiments.append({"name": "Tighten_Dedup_SMILES", **{**base, "deduplication": "Unique_SMILES"}})
    
    results = []
    best_r2 = -float("inf")
    best_name = None
    best_strategy = None
    
    for i, exp in enumerate(experiments):
        name = exp.pop("name")
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        
        try:
            df_clean = execute_strategy(df_raw, exp, action_space)
            n_clean = len(df_clean)
            retention = n_clean / len(df_raw) * 100
            
            if n_clean < 50:
                print(f"  {n_clean} rows ({retention:.2f}%) — too few, skipping")
                results.append({
                    "Exp": i+1, "Strategy": name, "N": n_clean, "Ret%": f"{retention:.1f}",
                    "Delta_R2": "N/A", "Delta_R2+Base": "N/A", "Abs_R2": "N/A", "RMSE": "N/A"
                })
                continue
            
            metrics = evaluate_strategy(df_clean)
            d_r2 = metrics["delta_pce_r2"]
            d_r2_base = metrics["delta_pce_r2_with_baseline"]
            abs_r2 = metrics["abs_pce_r2"]
            rmse = metrics["rmse_delta"]
            
            print(f"  {n_clean} rows ({retention:.2f}%) | ΔR²={d_r2:+.4f} | ΔR²+Base={d_r2_base:+.4f} | AbsR²={abs_r2:+.4f} | RMSE={rmse:.3f}")
            
            results.append({
                "Exp": i+1, "Strategy": name, "N": n_clean, "Ret%": f"{retention:.1f}",
                "Delta_R2": f"{d_r2:+.4f}", "Delta_R2+Base": f"{d_r2_base:+.4f}",
                "Abs_R2": f"{abs_r2:+.4f}", "RMSE": f"{rmse:.3f}"
            })
            
            if d_r2_base > best_r2:
                best_r2 = d_r2_base
                best_name = name
                best_strategy = exp.copy()
                print(f"  ★ NEW BEST!")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "Exp": i+1, "Strategy": name, "N": "ERR", "Ret%": "ERR",
                "Delta_R2": "ERR", "Delta_R2+Base": "ERR", "Abs_R2": "ERR", "RMSE": "ERR"
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("EVOLUTION SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    
    print(f"\nBEST: {best_name} | ΔR²+Base = {best_r2:+.4f}")
    
    # Save best
    if best_strategy:
        best_df = execute_strategy(df_raw, best_strategy, action_space)
        best_path = PROJECT_ROOT / "explorations" / "data_source_exploration" / "agentic_evolution_best.csv"
        best_df.to_csv(best_path, index=False)
        print(f"Saved best: {best_path} ({len(best_df)} rows)")
    
    # Save log
    log_path = LOG_DIR / "agentic_evolution_log.json"
    with open(log_path, "w") as f:
        json.dump({"best": {"name": best_name, "r2": best_r2, "strategy": best_strategy}, "results": results}, f, indent=2)
    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    run_evolution_from_baseline()
