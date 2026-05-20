#!/usr/bin/env python3
"""orchestrator.py

Multi-Agent Parallel Cross-Layer Exploration Orchestrator.

Spawns N WorkerAgents, each executing a random cross-layer pipeline combination.
Collects results, ranks pipelines, and saves a leaderboard.

Usage:
    python src/orchestrator.py --n-agents 20 --max-workers 4 --target delta_pce
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cross_layer_sampler import generate_unique_configs, _load_yaml, REGISTRY_PATH
from worker_agent import run_worker_star


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Cross-Layer Pipeline Exploration")
    parser.add_argument("--n-agents", type=int, default=20, help="Number of pipeline combinations to explore")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel processes (default: CPU count)")
    parser.add_argument("--target", type=str, default="delta_pce", choices=["delta_pce", "absolute_pce"],
                        help="Target variable to predict")
    parser.add_argument("--baseline-as-feature", action="store_true", default=True,
                        help="Include baseline PCE as input feature")
    parser.add_argument("--no-baseline-as-feature", action="store_false", dest="baseline_as_feature",
                        help="Do NOT include baseline PCE as input feature")
    parser.add_argument("--weighted-sampling", action="store_true", default=True,
                        help="Use weighted random sampling (priority to implemented/PSC-verified)")
    parser.add_argument("--uniform-sampling", action="store_false", dest="weighted_sampling",
                        help="Use uniform random sampling")
    parser.add_argument("--output", type=str, default="results/multi_agent_exploration",
                        help="Output directory for results")
    parser.add_argument("--fast-only", action="store_true", default=True,
                        help="Skip slow eval methods (Optuna, grid search)")
    parser.add_argument("--full-eval", action="store_false", dest="fast_only",
                        help="Allow slow eval methods (much longer runtime)")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save intermediate results every N agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def save_checkpoint(results: list, output_dir: Path, iteration: int):
    """Save intermediate results to disk."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_{iteration:04d}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Checkpoint] Saved {len(results)} results → {path}")


def build_leaderboard(results: list) -> pd.DataFrame:
    """Convert raw results to a ranked leaderboard DataFrame."""
    rows = []
    for r in results:
        cfg = r.get("config", {})
        metrics = r.get("metrics", {})
        row = {
            "agent_id": r.get("agent_id"),
            "status": r.get("status"),
            "duration_sec": r.get("duration_sec"),
            "n_samples": r.get("n_samples"),
            "n_features": r.get("n_features"),
            "r2": metrics.get("r2", -999),
            "r2_std": metrics.get("r2_std", 0),
            "rmse": metrics.get("rmse", 999),
            "L1": cfg.get("layer1", {}).get("method_id", "?"),
            "L2": cfg.get("layer2", {}).get("method_id", "?"),
            "L3": cfg.get("layer3", {}).get("method_id", "?"),
            "L4": cfg.get("layer4", {}).get("method_id", "?"),
            "L5": cfg.get("layer5", {}).get("method_id", "?"),
            "target": cfg.get("target", "?"),
            "baseline_feat": cfg.get("baseline_as_feature", False),
            "config_hash": cfg.get("_hash", ""),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # Sort by R² descending, but put errors at the bottom
    df["sort_key"] = df["r2"].where(df["status"] == "success", -9999)
    df = df.sort_values("sort_key", ascending=False).drop(columns=["sort_key"]).reset_index(drop=True)
    return df


def print_summary(leaderboard: pd.DataFrame, top_k: int = 10):
    """Print a console summary of the exploration results."""
    print("\n" + "=" * 80)
    print("MULTI-AGENT EXPLORATION SUMMARY")
    print("=" * 80)
    
    n_total = len(leaderboard)
    n_success = (leaderboard["status"] == "success").sum()
    n_error = n_total - n_success
    
    print(f"\nTotal agents:  {n_total}")
    print(f"Successful:    {n_success} ({100*n_success/n_total:.1f}%)")
    print(f"Failed:        {n_error} ({100*n_error/n_total:.1f}%)")
    
    if n_success > 0:
        success_df = leaderboard[leaderboard["status"] == "success"]
        print(f"\nR²  —  best: {success_df['r2'].max():+.4f}  |  mean: {success_df['r2'].mean():+.4f}  |  std: {success_df['r2'].std():.4f}")
        print(f"RMSE — best: {success_df['rmse'].min():.4f}  |  mean: {success_df['rmse'].mean():.4f}")
        print(f"Duration — mean: {success_df['duration_sec'].mean():.1f}s  |  total: {success_df['duration_sec'].sum():.1f}s")
        
        print(f"\n{'='*80}")
        print(f"TOP {top_k} PIPELINES (by R²)")
        print(f"{'='*80}")
        
        top = success_df.head(top_k)
        cols = ["rank", "r2", "rmse", "n_samples", "n_features", "L1", "L2", "L3", "L4", "duration_sec"]
        top_display = top[cols[1:]].copy()
        top_display.insert(0, "rank", range(1, len(top_display) + 1))
        print(top_display.to_string(index=False))
    
    print("\n" + "=" * 80)


def main():
    args = parse_args()
    
    # Setup
    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import random
    random.seed(args.seed)
    
    max_workers = args.max_workers or os.cpu_count()
    
    print("=" * 80)
    print("MULTI-AGENT PARALLEL CROSS-LAYER EXPLORATION")
    print("=" * 80)
    print(f"Agents:        {args.n_agents}")
    print(f"Max workers:   {max_workers}")
    print(f"Target:        {args.target}")
    print(f"Baseline feat: {args.baseline_as_feature}")
    print(f"Sampling:      {'weighted' if args.weighted_sampling else 'uniform'}")
    print(f"Fast only:     {args.fast_only}")
    print(f"Output dir:    {output_dir}")
    print(f"Started:       {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Phase 1: Generate configs
    print("\n[Phase 1] Generating pipeline configurations ...")
    registry = _load_yaml(REGISTRY_PATH)
    configs = generate_unique_configs(
        n=args.n_agents,
        registry=registry,
        target=args.target,
        baseline_as_feature=args.baseline_as_feature,
        weighted=args.weighted_sampling,
        fast_only=args.fast_only,
    )
    print(f"Generated {len(configs)} unique configurations.")
    
    # Phase 2: Prepare worker args
    worker_args = [(f"agent_{i+1:03d}", cfg) for i, cfg in enumerate(configs)]
    
    # Phase 3: Execute in parallel
    print(f"\n[Phase 2] Running {len(worker_args)} agents with {max_workers} workers ...")
    start_time = time.time()
    
    results = []
    if max_workers == 1:
        # Sequential execution (easier debugging)
        for agent_id, cfg in worker_args:
            print(f"  → {agent_id} | L1={cfg['layer1']['method_id']} L2={cfg['layer2']['method_id']} L3={cfg['layer3']['method_id']} L4={cfg['layer4']['method_id']}")
            res = run_worker_star((agent_id, cfg))
            results.append(res)
            if len(results) % args.checkpoint_every == 0:
                save_checkpoint(results, output_dir, len(results))
    else:
        # Parallel execution
        ctx = get_context("spawn")
        with ctx.Pool(processes=max_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(run_worker_star, worker_args), 1):
                results.append(res)
                status_icon = "✓" if res["status"] == "success" else "✗"
                print(f"  [{i}/{len(worker_args)}] {status_icon} {res['agent_id']} | R²={res.get('metrics', {}).get('r2', -999):+.4f} | {res['duration_sec']:.1f}s")
                if i % args.checkpoint_every == 0:
                    save_checkpoint(results, output_dir, i)
    
    total_duration = time.time() - start_time
    print(f"\nAll agents finished in {total_duration:.1f}s.")
    
    # Phase 4: Build leaderboard
    print("\n[Phase 3] Building leaderboard ...")
    leaderboard = build_leaderboard(results)
    
    # Phase 5: Save outputs
    leaderboard_csv = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_csv, index=False)
    print(f"Saved leaderboard → {leaderboard_csv}")
    
    results_json = output_dir / "all_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved all results → {results_json}")
    
    best_pipelines_json = output_dir / "best_pipelines.json"
    best = leaderboard[leaderboard["status"] == "success"].head(10)
    best.to_json(best_pipelines_json, orient="records", indent=2)
    print(f"Saved best pipelines → {best_pipelines_json}")
    
    # Phase 6: Print summary
    print_summary(leaderboard, top_k=10)
    
    # Save run metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "total_agents": args.n_agents,
        "successful": int((leaderboard["status"] == "success").sum()),
        "failed": int((leaderboard["status"] != "success").sum()),
        "best_r2": float(leaderboard[leaderboard["status"] == "success"]["r2"].max()) if (leaderboard["status"] == "success").any() else -999,
        "total_duration_sec": round(total_duration, 2),
    }
    meta_path = output_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {meta_path}")


if __name__ == "__main__":
    main()
