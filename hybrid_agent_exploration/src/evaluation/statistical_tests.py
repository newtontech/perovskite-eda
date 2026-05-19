"""statistical_tests.py — Statistical significance testing for model comparison.

Provides Friedman test + Nemenyi post-hoc test for comparing multiple models
across repeated evaluations, and paired permutation tests for head-to-head
model comparisons.
"""

import json
import warnings
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")


def friedman_nemenyi_test(results: list[dict], metric_key: str = "r2") -> dict[str, Any]:
    """Friedman test + Nemenyi post-hoc test for comparing multiple models.

    Supports two input modes:
      1. Repeated independent evaluations (each config has multiple metric values)
      2. CV fold scores extracted from artifacts (each config has 5+ fold scores)

    Args:
        results: List of result dicts, each with 'metrics' containing metric_key.
        metric_key: Metric to compare (default 'r2').

    Returns:
        Dict with friedman_statistic, friedman_pvalue, nemenyi_matrix, and
        pairwise comparisons.
    """
    from scipy.stats import friedmanchisquare

    # --- Strategy: group by validation strategy, then compare models within each group ---
    # We run separate Friedman tests for each L4 strategy (random_split, 5fold_cv, etc.)
    from pathlib import Path
    # Determine artifact directory relative to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]  # src/evaluation/ -> src/ -> root
    artifact_dir = project_root / "results" / "report_artifacts"

    # Group results by L4 strategy
    by_strategy = {}
    for r in results:
        l4 = r.get("config", {}).get("layer4", {}).get("method_id", "unknown")
        by_strategy.setdefault(l4, []).append(r)

    overall = {
        "tests_by_strategy": {},
        "summary": {},
    }

    for strategy, strat_results in by_strategy.items():
        grouped = {}
        for r in strat_results:
            cfg = r.get("config", {})
            # Use L2+L3 as model identifier (strip L4 and L5)
            model_id = f"{cfg.get('layer2', {}).get('method_id', '?')}_{cfg.get('layer3', {}).get('method_id', '?')}"
            agent_id = r.get("agent_id", "")
            # Try to load artifact file for CV fold scores
            cv_scores = None
            if agent_id:
                art_path = artifact_dir / f"{agent_id}_artifacts.json"
                if art_path.exists():
                    try:
                        with open(art_path) as fh:
                            art = json.load(fh)
                        cv_scores = art.get("cv_scores_per_fold")
                    except Exception as e:
                        pass
            # DEBUG
            if agent_id and strategy == "E43_5fold_cv":
                has = art_path.exists() if agent_id else False
                n_scores = len(cv_scores) if cv_scores else 0
            if cv_scores and len(cv_scores) >= 2:
                grouped[model_id] = [float(v) for v in cv_scores]
                continue
            # Fallback: use the single metric value
            val = r.get("metrics", {}).get(metric_key)
            if val is not None:
                grouped.setdefault(model_id, []).append(float(val))


        if len(grouped) < 3:
            overall["tests_by_strategy"][strategy] = {
                "error": "Need at least 3 distinct models for Friedman test",
                "n_models": len(grouped),
            }
            continue

        min_len = min(len(v) for v in grouped.values())
        if min_len < 2:
            overall["tests_by_strategy"][strategy] = {
                "error": "Need at least 2 repeated evaluations (or CV folds) per model",
                "min_repeats": min_len,
            }
            continue

        names = list(grouped.keys())
        data_matrix = np.array([grouped[n][:min_len] for n in names])

        try:
            stat, pvalue = friedmanchisquare(*data_matrix)
        except Exception as e:
            overall["tests_by_strategy"][strategy] = {"error": f"Friedman test failed: {e}"}
            continue

        k = len(names)
        N = min_len
        q_alpha_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                         7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
        q_alpha = q_alpha_table.get(k, 2.728)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

        means = {n: np.mean(grouped[n][:min_len]) for n in names}
        pairwise = []
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                diff = abs(means[n1] - means[n2])
                pairwise.append({
                    "model_a": n1,
                    "model_b": n2,
                    "mean_diff": float(diff),
                    "significant": bool(diff > cd),
                    "cd_threshold": float(cd),
                })

        overall["tests_by_strategy"][strategy] = {
            "friedman_statistic": float(stat),
            "friedman_pvalue": float(pvalue),
            "nemenyi_critical_difference": float(cd),
            "n_models": k,
            "n_repeats": N,
            "model_means": {n: float(v) for n, v in means.items()},
            "pairwise_comparisons": pairwise,
            "significant_difference": bool(pvalue < 0.05),
            "data_source": "cv_fold_scores" if any(len(v) >= 3 for v in grouped.values()) else "single_evaluations",
        }

    # Overall summary: aggregate across strategies
    n_significant = sum(1 for t in overall["tests_by_strategy"].values()
                        if t.get("significant_difference"))
    n_tests = len([t for t in overall["tests_by_strategy"].values() if "friedman_pvalue" in t])
    overall["summary"] = {
        "n_strategies_tested": len(by_strategy),
        "n_friedman_tests_run": n_tests,
        "n_significant": n_significant,
        "any_significant": n_significant > 0,
    }
    return overall


def paired_permutation_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    metric_fn: callable = None,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Paired permutation test to compare two models' predictions.

    Args:
        y_true: Ground truth values.
        preds_a: Predictions from model A.
        preds_b: Predictions from model B.
        metric_fn: Function(y_true, y_pred) -> score. Default: negative RMSE.
        n_permutations: Number of permutation iterations.
        random_state: Random seed.

    Returns:
        Dict with observed_diff, pvalue, effect_size, and permutation_distribution.
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    if metric_fn is None:
        def metric_fn(y_t, y_p):
            return -np.sqrt(np.mean((y_t - y_p) ** 2))  # negative RMSE (higher is better)

    score_a = metric_fn(y_true, preds_a)
    score_b = metric_fn(y_true, preds_b)
    observed_diff = score_a - score_b

    # Permutation: randomly swap predictions for each sample
    diffs = []
    for _ in range(n_permutations):
        swap = rng.rand(len(y_true)) < 0.5
        perm_a = np.where(swap, preds_b, preds_a)
        perm_b = np.where(swap, preds_a, preds_b)
        diffs.append(metric_fn(y_true, perm_a) - metric_fn(y_true, perm_b))

    diffs = np.array(diffs)
    pvalue = np.mean(np.abs(diffs) >= np.abs(observed_diff))

    # Effect size: Cohen's d for paired samples
    diff_scores = (preds_a - y_true) ** 2 - (preds_b - y_true) ** 2
    cohens_d = float(np.mean(diff_scores) / (np.std(diff_scores, ddof=1) + 1e-10))

    return {
        "observed_diff": float(observed_diff),
        "pvalue": float(pvalue),
        "effect_size_cohens_d": cohens_d,
        "n_permutations": n_permutations,
        "metric": "negative_RMSE",
        "model_a_better": observed_diff > 0 and pvalue < 0.05,
        "model_b_better": observed_diff < 0 and pvalue < 0.05,
    }


def compute_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    method: str = "bootstrap",
    n_bootstrap: int = 10000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute confidence interval for a metric using bootstrap.

    Args:
        scores: List of scores (e.g., fold R² values).
        confidence: Confidence level (default 0.95).
        method: "bootstrap" or "t_interval".
        n_bootstrap: Number of bootstrap samples.
        random_state: Random seed.

    Returns:
        Dict with mean, std, ci_lower, ci_upper.
    """
    scores = np.array(scores)
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))

    if method == "bootstrap":
        rng = np.random.RandomState(random_state)
        boot_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(scores, size=len(scores), replace=True)
            boot_means.append(np.mean(sample))
        boot_means = np.array(boot_means)
        alpha = 1 - confidence
        ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    else:
        from scipy.stats import t
        alpha = 1 - confidence
        t_val = t.ppf(1 - alpha / 2, df=len(scores) - 1)
        margin = t_val * std / np.sqrt(len(scores))
        ci_lower = mean - margin
        ci_upper = mean + margin

    return {
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "method": method,
        "n_samples": len(scores),
    }
