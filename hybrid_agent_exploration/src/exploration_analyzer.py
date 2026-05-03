#!/usr/bin/env python3
"""exploration_analyzer.py

Post-run analysis for multi-agent cross-layer exploration results.

Usage:
    python src/exploration_analyzer.py --input results/multi_agent_exploration
"""

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def load_results(input_dir: Path) -> pd.DataFrame:
    """Load leaderboard CSV from an exploration run."""
    csv_path = input_dir / "leaderboard.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {csv_path}")
    return pd.read_csv(csv_path)


def analyze_layer_contributions(df: pd.DataFrame, top_k: int = 10) -> dict:
    """Analyze which methods per layer appear most often in top pipelines."""
    success = df[df["status"] == "success"]
    if success.empty:
        return {}
    
    top = success.head(top_k)
    analysis = {}
    for layer_col in ["L1", "L2", "L3", "L4", "L5"]:
        counts = Counter(top[layer_col])
        analysis[layer_col] = dict(counts.most_common())
    return analysis


def analyze_failure_modes(df: pd.DataFrame) -> dict:
    """Analyze which layer combinations fail most often."""
    failures = df[df["status"] != "success"]
    if failures.empty:
        return {"note": "No failures recorded"}
    
    analysis = {
        "total_failures": len(failures),
        "by_L3_model": Counter(failures["L3"]).most_common(),
        "by_L2_feature": Counter(failures["L2"]).most_common(),
        "by_L1_cleaning": Counter(failures["L1"]).most_common(),
    }
    return analysis


def analyze_r2_vs_complexity(df: pd.DataFrame) -> dict:
    """Correlation between n_features and R²."""
    success = df[df["status"] == "success"]
    if success.empty or len(success) < 3:
        return {"note": "Insufficient data for correlation"}
    
    corr = success[["n_features", "r2", "rmse", "duration_sec"]].corr()
    return {
        "n_features_vs_r2": float(corr.loc["n_features", "r2"]),
        "n_features_vs_rmse": float(corr.loc["n_features", "rmse"]),
        "duration_vs_r2": float(corr.loc["duration_sec", "r2"]),
    }


def generate_report(df: pd.DataFrame) -> dict:
    """Generate a comprehensive analysis report."""
    report = {
        "overview": {
            "total_agents": int(len(df)),
            "successful": int((df["status"] == "success").sum()),
            "failed": int((df["status"] != "success").sum()),
            "best_r2": float(df[df["status"] == "success"]["r2"].max()) if (df["status"] == "success").any() else None,
            "worst_r2": float(df[df["status"] == "success"]["r2"].min()) if (df["status"] == "success").any() else None,
            "mean_r2": float(df[df["status"] == "success"]["r2"].mean()) if (df["status"] == "success").any() else None,
        },
        "top_10_layer_contributions": analyze_layer_contributions(df, top_k=10),
        "failure_analysis": analyze_failure_modes(df),
        "complexity_analysis": analyze_r2_vs_complexity(df),
    }
    return report


def save_report(report: dict, output_dir: Path):
    """Save analysis report to JSON."""
    path = output_dir / "analysis_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved analysis report → {path}")


def print_report(report: dict):
    """Print analysis report to console."""
    print("\n" + "=" * 80)
    print("EXPLORATION ANALYSIS REPORT")
    print("=" * 80)
    
    ov = report["overview"]
    print(f"\nOverview:")
    print(f"  Total agents:    {ov['total_agents']}")
    print(f"  Successful:      {ov['successful']} ({100*ov['successful']/ov['total_agents']:.1f}%)")
    print(f"  Failed:          {ov['failed']}")
    if ov['best_r2'] is not None:
        print(f"  Best R²:         {ov['best_r2']:+.4f}")
        print(f"  Worst R²:        {ov['worst_r2']:+.4f}")
        print(f"  Mean R²:         {ov['mean_r2']:+.4f}")
    
    contrib = report.get("top_10_layer_contributions", {})
    if contrib:
        print(f"\nTop-10 Pipeline Layer Composition:")
        for layer, counts in contrib.items():
            print(f"  {layer}: {dict(counts)}")
    
    fail = report.get("failure_analysis", {})
    if fail.get("total_failures"):
        print(f"\nFailure Analysis ({fail['total_failures']} failures):")
        print(f"  Most failing models:      {fail['by_L3_model'][:3]}")
        print(f"  Most failing features:    {fail['by_L2_feature'][:3]}")
    
    comp = report.get("complexity_analysis", {})
    if "n_features_vs_r2" in comp:
        print(f"\nComplexity vs Performance:")
        print(f"  n_features ↔ R²:   {comp['n_features_vs_r2']:+.3f}")
        print(f"  n_features ↔ RMSE: {comp['n_features_vs_rmse']:+.3f}")
        print(f"  duration ↔ R²:     {comp['duration_vs_r2']:+.3f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze Multi-Agent Exploration Results")
    parser.add_argument("--input", type=str, required=True, help="Directory containing exploration outputs")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    df = load_results(input_dir)
    
    report = generate_report(df)
    save_report(report, input_dir)
    print_report(report)


if __name__ == "__main__":
    main()
