#!/usr/bin/env python3
"""generate_research_story.py

Unified Research Story Generator — runs experiments AND generates reports.

Workflow:
  1. Define a systematic experiment matrix (e.g., 3 algorithms × 4 features)
  2. Run all experiments (parallel), saving rich artifacts
  3. Aggregate results and artifacts
  4. Generate top-journal main-text report (5-8 composite figures)
  5. Generate Supporting Information (10-30+ detailed figures and tables)

Usage:
    # Run a full research story (experiment + report)
    python src/generate_research_story.py \
      --matrix jpcl_sam_matrix \
      --output results/research_stories/story_001

    # Generate report from existing results
    python src/generate_research_story.py \
      --input results/multi_agent_test \
      --output results/research_stories/story_001 \
      --skip-experiments
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

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reporting.top_journal_report import TopJournalReport
from reporting.si_generator import SIGenerator
from reporting.figure_selector import FigureSelector
from worker_agent import run_worker_star


# ---------------------------------------------------------------------------
# Pre-defined experiment matrices
# ---------------------------------------------------------------------------

MATRIX_DEFINITIONS = {
    "jpcl_sam_matrix": {
        "description": "Systematic matrix: 3 algorithms × 4 feature representations (JPCL SAM-style)",
        "layer1": ["agentic_veryloose"],
        "layer2": ["F21_rdkit_basic", "F22_morgan_256", "F22_morgan_512", "F22_morgan_1024"],
        "layer3": ["M31_random_forest", "M31_xgboost", "M31_gradient_boosting"],
        "layer4": ["E42_random_split"],
        "layer5": ["D54_report_only"],
        "baseline_as_feature": True,
        "target": "delta_pce",
    },
    "jpcl_sam_matrix_v2": {
        "description": "Enhanced matrix: 3 algorithms × 4 features × 3 validation strategies + Optuna + statistical tests",
        "layer1": ["agentic_veryloose"],
        "layer2": ["F21_rdkit_basic", "F22_morgan_256", "F22_morgan_512", "F22_morgan_1024"],
        "layer3": ["M31_random_forest", "M31_xgboost", "M31_gradient_boosting"],
        "layer4": ["E42_random_split", "E43_5fold_cv", "E42_scaffold_split"],
        "layer5": ["D54_report_only"],
        "baseline_as_feature": True,
        "target": "delta_pce",
        "auto_optuna_for_top_k": 3,
    },
    "minimal_test": {
        "description": "Minimal 2×2 matrix for quick testing",
        "layer1": ["agentic_veryloose"],
        "layer2": ["F21_rdkit_basic", "F22_maccs"],
        "layer3": ["M31_random_forest", "M31_xgboost"],
        "layer4": ["E42_random_split"],
        "layer5": ["D54_report_only"],
        "baseline_as_feature": True,
        "target": "delta_pce",
    },
}


def build_matrix_configs(matrix_def: dict) -> list[dict]:
    """Generate all combinations from a matrix definition."""
    import itertools
    configs = []
    for l1, l2, l3, l4, l5 in itertools.product(
        matrix_def["layer1"],
        matrix_def["layer2"],
        matrix_def["layer3"],
        matrix_def["layer4"],
        matrix_def["layer5"],
    ):
        cfg = {
            "layer1": {"method_id": l1, "strategy": l1 if l1.startswith("agentic") else l1},
            "layer2": {"method_id": l2},
            "layer3": {"method_id": l3},
            "layer4": {"method_id": l4},
            "layer5": {"method_id": l5},
            "target": matrix_def["target"],
            "baseline_as_feature": matrix_def["baseline_as_feature"],
            "_hash": f"{l1}_{l2}_{l3}_{l4}_{l5}",
        }
        configs.append(cfg)
    return configs


def run_experiments(configs: list[dict], max_workers: int | None = None,
                    output_dir: Path | None = None) -> list[dict]:
    """Run all experiments in the matrix, parallel if possible."""
    max_workers = max_workers or min(os.cpu_count(), 4)
    worker_args = [(f"agent_{i+1:03d}", cfg) for i, cfg in enumerate(configs)]

    print(f"Running {len(worker_args)} experiments with {max_workers} workers...")
    start = time.time()
    results = []

    if max_workers == 1:
        for agent_id, cfg in worker_args:
            res = run_worker_star((agent_id, cfg))
            results.append(res)
            status = "✓" if res["status"] == "success" else "✗"
            print(f"  [{len(results)}/{len(worker_args)}] {status} {agent_id} | R²={res.get('metrics', {}).get('r2', -999):+.4f}")
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=max_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(run_worker_star, worker_args), 1):
                results.append(res)
                status = "✓" if res["status"] == "success" else "✗"
                print(f"  [{i}/{len(worker_args)}] {status} {res['agent_id']} | R²={res.get('metrics', {}).get('r2', -999):+.4f} | {res['duration_sec']:.1f}s")

    print(f"All experiments finished in {time.time() - start:.1f}s.")
    return results


def save_experiment_outputs(results: list[dict], output_dir: Path):
    """Save results, leaderboard, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    # Save all results
    results_path = output_dir / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results → {results_path}")

    # Build and save leaderboard
    rows = []
    for r in results:
        cfg = r.get("config", {})
        m = r.get("metrics", {})
        rows.append({
            "agent_id": r.get("agent_id"),
            "status": r.get("status"),
            "r2": m.get("r2", -999),
            "rmse": m.get("rmse", 999),
            "mae": m.get("mae", 999),
            "pearson_r": m.get("pearson_r", -999),
            "L1": cfg.get("layer1", {}).get("method_id", "?"),
            "L2": cfg.get("layer2", {}).get("method_id", "?"),
            "L3": cfg.get("layer3", {}).get("method_id", "?"),
            "L4": cfg.get("layer4", {}).get("method_id", "?"),
            "n_samples": r.get("n_samples", 0),
            "n_features": r.get("n_features", 0),
        })
    df = pd.DataFrame(rows)
    df["sort_key"] = df["r2"].where(df["status"] == "success", -9999)
    df = df.sort_values("sort_key", ascending=False).drop(columns=["sort_key"]).reset_index(drop=True)
    leaderboard_path = output_dir / "leaderboard.csv"
    df.to_csv(leaderboard_path, index=False)
    print(f"Saved leaderboard → {leaderboard_path}")


def load_results(input_dir: Path) -> list[dict]:
    path = input_dir / "all_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_artifacts(artifacts_dir: Path, results: list[dict] | None = None) -> dict:
    """Aggregate all artifact files into a single dict.

    Builds two views:
      1. Per-agent nested dict for multi-model comparison
      2. Flattened best-model artifacts for single-model figures
      3. multi_model_results list for comparison figures
    """
    artifacts = {}
    per_agent = {}
    if not artifacts_dir.exists():
        return artifacts

    for f in sorted(artifacts_dir.glob("*_artifacts.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            agent_id = f.stem.replace("_artifacts", "")
            per_agent[agent_id] = data
        except Exception:
            pass

    if not per_agent:
        return artifacts

    # Build multi_model_results list
    multi_model_results = []
    # Map agent_id to result metrics for enrichment
    result_map = {r["agent_id"]: r for r in (results or [])}

    for agent_id, data in per_agent.items():
        res = result_map.get(agent_id, {})
        cfg = res.get("config", {})
        m = res.get("metrics", {})
        model_entry = {
            "name": f"{cfg.get('layer3', {}).get('method_id', '?')} + {cfg.get('layer2', {}).get('method_id', '?')}",
            "agent_id": agent_id,
            "r2": m.get("r2", -999),
            "rmse": m.get("rmse", 999),
            "mae": m.get("mae", 999),
            "pearson_r": m.get("pearson_r", -999),
            "n_samples": res.get("n_samples", 0),
            "n_features": res.get("n_features", 0),
        }
        # Copy artifact keys
        for key in ["y_true", "y_pred", "feature_importances", "cv_scores_per_fold",
                    "shap_values", "shap_background", "feature_names"]:
            if key in data:
                model_entry[key] = data[key]
        multi_model_results.append(model_entry)

    # Sort by R² descending
    multi_model_results.sort(key=lambda x: x.get("r2", -999), reverse=True)
    artifacts["multi_model_results"] = multi_model_results

    # Use best model's artifacts as flat fallback for single-model figures
    if multi_model_results:
        best = multi_model_results[0]
        for key in ["y_true", "y_pred", "feature_importances", "cv_scores_per_fold",
                    "shap_values", "shap_background", "feature_names"]:
            if key in best:
                artifacts[key] = best[key]

    # Also keep per-agent dict for advanced selectors
    artifacts["_per_agent"] = per_agent

    return artifacts


def generate_reports(
    results: list[dict],
    artifacts: dict,
    output_dir: Path,
    *,
    embed_images: bool = False,
    quality_target: str = "standard",
    quality_gate: str = "standard",
    journal_profile: str = "jpcl",
    require_external_validation: bool = False,
):
    """Generate main text + SI reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if require_external_validation and not (
        artifacts.get("y_true_external") and artifacts.get("y_pred_external")
    ):
        raise ValueError(
            "--require-external-validation was set, but no external validation artifacts were found"
        )

    # Print figure generation plan
    selector = FigureSelector(artifacts)
    print("\n" + selector.generate_report_summary())

    # Main text
    print("\nGenerating main-text report...")
    main_report = TopJournalReport(
        results=results, artifacts=artifacts,
        output_dir=output_dir / "main_text",
        quality_target=quality_target,
        embed_images=embed_images,
    )
    main_bundle = main_report.generate()
    gate_thresholds = {"draft": 0.5, "standard": 0.7, "strict": 0.85}
    threshold = gate_thresholds[quality_gate]
    if main_bundle.quality_score < threshold:
        raise RuntimeError(
            f"Report quality gate failed for {journal_profile}: "
            f"{main_bundle.quality_score:.3f} < {threshold:.3f}"
        )
    print(f"Main text → {main_bundle.path} (quality_score={main_bundle.quality_score:.3f})")

    # SI
    print("\nGenerating Supporting Information...")
    si = SIGenerator(
        results=results, artifacts=artifacts,
        output_dir=output_dir / "si",
        embed_images=embed_images,
    )
    si_path = si.generate()
    print(f"SI → {si_path}")

    return main_bundle, si_path


def main():
    parser = argparse.ArgumentParser(description="Unified Research Story Generator")
    parser.add_argument("--matrix", type=str, default="minimal_test",
                        choices=list(MATRIX_DEFINITIONS.keys()),
                        help="Experiment matrix to run")
    parser.add_argument("--input", type=str, default=None,
                        help="Existing results directory (skip experiments if provided)")
    parser.add_argument("--output", type=str, default="results/research_stories/story_001",
                        help="Output directory")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers (default: min(CPU, 4))")
    parser.add_argument("--skip-experiments", action="store_true",
                        help="Skip experiment running, generate report from --input only")
    parser.add_argument("--only-report", action="store_true",
                        help="Alias for --skip-experiments")
    parser.add_argument("--embed-images", action="store_true",
                        help="Embed local images as base64 data URIs in generated markdown")
    parser.add_argument("--quality-target", type=str, default="standard",
                        choices=["standard", "top-journal"],
                        help="Target report quality profile")
    parser.add_argument("--quality-gate", type=str, default="standard",
                        choices=["draft", "standard", "strict"],
                        help="Minimum automated report quality gate")
    parser.add_argument("--journal-profile", type=str, default="jpcl",
                        choices=["jpcl", "nature-energy", "joule", "ees", "custom"],
                        help="Journal style profile for quality-gate messaging")
    parser.add_argument("--run-review", action="store_true",
                        help="Print reviewer and claim-audit sidecar paths after generation")
    parser.add_argument("--resume-manifest", type=str, default=None,
                        help="Existing run_manifest.json to carry forward as prior state")
    parser.add_argument("--require-external-validation", action="store_true",
                        help="Fail if independent external-validation artifacts are missing")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Run experiments (unless skipped)
    if args.skip_experiments or args.only_report:
        if not args.input:
            raise ValueError("--input is required when skipping experiments")
        input_dir = Path(args.input)
        print(f"Skipping experiments. Loading existing results from: {input_dir}")
        results = load_results(input_dir)
    else:
        matrix_def = MATRIX_DEFINITIONS[args.matrix]
        print(f"Matrix: {args.matrix}")
        print(f"Description: {matrix_def['description']}")
        configs = build_matrix_configs(matrix_def)
        print(f"Total configurations: {len(configs)}")

        # Clean old artifacts
        artifacts_dir = PROJECT_ROOT / "results" / "report_artifacts"
        if artifacts_dir.exists():
            for f in artifacts_dir.glob("*_artifacts.json"):
                f.unlink()
            print("Cleared old artifacts.")

        results = run_experiments(configs, max_workers=args.max_workers)
        input_dir = output_dir / "experiments"
        save_experiment_outputs(results, input_dir)

    # --- Statistical tests ---
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "evaluation"))
        from statistical_tests import friedman_nemenyi_test
        stat_results = friedman_nemenyi_test(results, metric_key="r2")
        stat_path = input_dir / "statistical_tests.json"
        with open(stat_path, "w") as f:
            json.dump(stat_results, f, indent=2, default=str)
        print(f"Saved statistical tests → {stat_path}")
        if stat_results.get("significant_difference"):
            print(f"  Friedman p-value: {stat_results['friedman_pvalue']:.4f} (significant)")
        else:
            print(f"  Friedman p-value: {stat_results.get('friedman_pvalue', 'N/A')} (not significant or insufficient data)")
    except Exception as e:
        print(f"Statistical tests skipped: {e}")

    # --- Auto Optuna for top-k configurations ---
    matrix_def = MATRIX_DEFINITIONS.get(args.matrix, {})
    top_k = matrix_def.get("auto_optuna_for_top_k", 0)
    if top_k > 0 and not (args.skip_experiments or args.only_report):
        print(f"\n--- Auto Optuna: optimizing top-{top_k} configurations ---")
        successful = [r for r in results if r.get("status") == "success" and "r2" in r.get("metrics", {})]
        successful.sort(key=lambda r: r["metrics"]["r2"], reverse=True)
        for rank, res in enumerate(successful[:top_k], 1):
            cfg = res.get("config", {}).copy()
            cfg["layer4"] = {"method_id": "E44_optuna"}
            cfg["_hash"] = cfg.get("_hash", "") + "_optuna"
            agent_id = f"optuna_{rank:03d}"
            print(f"  Running Optuna for {agent_id} (base: {res['agent_id']}, R²={res['metrics']['r2']:.4f})...")
            try:
                from worker_agent import run_worker
                optuna_res = run_worker(agent_id, cfg)
                results.append(optuna_res)
                if optuna_res["status"] == "success":
                    print(f"    ✓ Optuna result: R²={optuna_res['metrics'].get('r2', -999):.4f}, best_params={optuna_res['metrics'].get('best_params', 'N/A')}")
                else:
                    print(f"    ✗ Optuna failed: {optuna_res.get('error', 'Unknown')}")
            except Exception as e:
                print(f"    ✗ Optuna error: {e}")
        # Re-save with Optuna results
        save_experiment_outputs(results, input_dir)

    print(f"\nLoaded {len(results)} experiment results")

    # Phase 2: Load artifacts
    artifacts_dir = PROJECT_ROOT / "results" / "report_artifacts"
    artifacts = load_artifacts(artifacts_dir, results=results)
    if args.resume_manifest:
        resume_path = Path(args.resume_manifest)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume manifest not found: {resume_path}")
        with open(resume_path) as f:
            artifacts["_resume_manifest"] = json.load(f)
        print(f"Loaded resume manifest → {resume_path}")
    print(f"Loaded artifacts with keys: {list(artifacts.keys())}")

    # Guardrail: deduplicate references before report generation
    from harness.guardrail import Guardrail
    guardrail = Guardrail()
    all_refs = []
    for r in results:
        cfg = r.get("config", {})
        if "references" in cfg:
            all_refs.extend(cfg["references"])
    if all_refs:
        all_refs = guardrail.check_duplicate_references(all_refs)
        print(f"Guardrail: deduplicated to {len(all_refs)} unique references")
    else:
        print("Guardrail: no references to deduplicate")

    # Phase 3: Generate reports
    main_bundle, si_path = generate_reports(
        results,
        artifacts,
        output_dir,
        embed_images=args.embed_images,
        quality_target=args.quality_target,
        quality_gate=args.quality_gate,
        journal_profile=args.journal_profile,
        require_external_validation=args.require_external_validation,
    )

    print("\n" + "=" * 60)
    print("Research Story generation complete.")
    print(f"Main text:  {main_bundle.path}")
    print(f"SI:         {si_path}")
    print(f"Manifest:   {main_bundle.manifest_path}")
    print(f"Claims:     {main_bundle.path.parent / 'claim_ledger.json'}")
    if args.run_review:
        print(f"Review:     {main_bundle.review_path}")
        print(f"Warnings:   {len(main_bundle.warnings)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
