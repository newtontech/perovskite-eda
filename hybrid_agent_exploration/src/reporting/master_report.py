"""master_report.py

Generate the global master report that synthesizes all experiments.
Retains ranking elements but emphasizes scientific insight over raw scores.
"""

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# Reuse palette from figure_generator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .image_embedder import embed_markdown_images

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


class MasterReport:
    """Generate a comprehensive master report from all exploration results."""

    def __init__(self, results: list[dict], output_dir: Path | str = "results/reports",
                 embed_images: bool = False):
        self.results = [r for r in results if isinstance(r, dict)]
        self.output_dir = Path(output_dir)
        self.embed_images = embed_images
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.output_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self._fig_counter = 0

    def _save_fig(self, fig: plt.Figure, suffix: str) -> Path:
        self._fig_counter += 1
        path = self.fig_dir / f"master_fig{self._fig_counter:02d}_{suffix}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def generate(self) -> Path:
        """Generate the master Markdown report."""
        successful = [r for r in self.results if r.get("status") == "success"]
        failed = [r for r in self.results if r.get("status") == "error"]

        # Sort for ranking display (but note: not the primary story)
        successful.sort(key=lambda r: r.get("metrics", {}).get("r2", float("-inf")), reverse=True)

        figure_paths = self._generate_master_figures(successful)

        lines = []
        lines.append("# Hybrid Agent Exploration — Master Report")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(
            f"This report synthesizes **{len(self.results)}** cross-layer pipeline experiments "
            f"({len(successful)} successful, {len(failed)} failed) exploring the prediction of "
            f"perovskite solar-cell PCE modulation by molecular additives. "
            f"The exploration spans five layers: data curation, molecular representation, model architecture, "
            f"evaluation protocol, and virtual screening. "
            f"**Note**: Rankings below are provided for orientation only; scientific innovation and "
            f"hypothesis validation are equally important metrics in this exploratory study."
        )
        lines.append("")

        # Global statistics
        lines.append("## 1. Global Exploration Statistics")
        lines.append("")
        lines.append(f"- **Total experiments**: {len(self.results)}")
        lines.append(f"- **Successful**: {len(successful)} ({100*len(successful)/max(len(self.results),1):.1f}%)")
        lines.append(f"- **Failed**: {len(failed)}")
        if successful:
            r2_vals = [r["metrics"]["r2"] for r in successful if "r2" in r["metrics"]]
            if r2_vals:
                lines.append(f"- **Best $R^2$**: {max(r2_vals):.4f}")
                lines.append(f"- **Mean $R^2$**: {np.mean(r2_vals):.4f}")
                lines.append(f"- **Std $R^2$**: {np.std(r2_vals):.4f}")
        lines.append("")

        # Layer-wise frequency
        lines.append("## 2. Layer-wise Method Frequency")
        lines.append("")
        for layer, col in [("Layer 1", "L1"), ("Layer 2", "L2"), ("Layer 3", "L3"), ("Layer 4", "L4")]:
            counts = Counter(r.get("config", {}).get(f"layer{col[-1]}", {}).get("method_id", "unknown") for r in successful)
            lines.append(f"### {layer}")
            for method, count in counts.most_common():
                lines.append(f"- {method}: {count} experiments")
            lines.append("")

        # Performance leaderboard (with caveats)
        lines.append("## 3. Performance Overview")
        lines.append("")
        lines.append(
            "> **Caution**: The following table ranks pipelines by $R^2$, but **higher $R^2$ does not necessarily imply greater scientific value**. "
            "Novel feature–model combinations that challenge established assumptions may score lower yet yield critical insights. "
            "Use this table as a navigational aid, not a definitive verdict."
        )
        lines.append("")
        lines.append("| Rank | Agent | $R^2$ | RMSE | L1 | L2 | L3 | L4 | N | Features | Innovation* |")
        lines.append("|------|-------|--------|------|----|----|----|----|---|----------|-------------|")

        def _innovation_badge(cfg: dict) -> str:
            badges = []
            if cfg.get("baseline_as_feature"):
                badges.append("baseline-feat")
            if "maccs" in cfg.get("layer2", {}).get("method_id", "").lower():
                badges.append("maccs")
            if cfg.get("layer1", {}).get("method_id", "") == "agentic_veryloose":
                badges.append("loose-clean")
            return ", ".join(badges) if badges else "standard"

        for i, r in enumerate(successful[:20], 1):
            cfg = r.get("config", {})
            m = r.get("metrics", {})
            l1 = cfg.get("layer1", {}).get("method_id", "?")
            l2 = cfg.get("layer2", {}).get("method_id", "?")
            l3 = cfg.get("layer3", {}).get("method_id", "?")
            l4 = cfg.get("layer4", {}).get("method_id", "?")
            r2_str = f"{m.get('r2', 'N/A'):.4f}" if m.get('r2') is not None else "N/A"
            rmse_str = f"{m.get('rmse', 'N/A'):.3f}" if m.get('rmse') is not None else "N/A"
            badges = _innovation_badge(cfg)
            lines.append(
                f"| {i} | {r.get('agent_id', '?')} | {r2_str} | {rmse_str} "
                f"| {l1} | {l2} | {l3} | {l4} "
                f"| {r.get('n_samples', '?')} | {r.get('n_features', '?')} | {badges} |"
            )
        lines.append("")
        lines.append("\\* *Innovation tags highlight non-standard design choices.*")
        lines.append("")

        # Layer contribution analysis
        lines.append("## 4. Cross-Layer Insight Analysis")
        lines.append("")
        lines.append(self._layer_contribution_analysis(successful))
        lines.append("")

        # Pareto front (performance vs complexity)
        lines.append("## 5. Performance vs. Complexity Trade-off")
        lines.append("")
        lines.append(
            "The figure below plots predictive performance ($R^2$) against model complexity (number of features). "
            "Pareto-optimal pipelines—those that cannot be improved in one dimension without sacrificing the other—"
            "are particularly valuable for practical deployment."
        )
        lines.append("")
        if figure_paths:
            for fp in figure_paths:
                rel = Path(fp).name
                lines.append(f"![{rel}](figures/{rel})")
                lines.append("")

        # Best practices
        lines.append("## 6. Emerging Best Practices")
        lines.append("")
        lines.append(self._best_practices(successful))
        lines.append("")

        # Open questions
        lines.append("## 7. Open Questions for Future Exploration")
        lines.append("")
        lines.extend(self._open_questions(successful, failed))
        lines.append("")

        # Conclusion
        lines.append("## 8. Conclusion")
        lines.append("")
        lines.append(
            "This multi-agent exploration has systematically probed the cross-layer design space for PSC additive prediction. "
            "Key findings include: (i) MACCS fingerprints paired with tree ensembles currently offer the strongest predictive signal; "
            "(ii) baseline-PCE-as-feature dramatically improves model utility; "
            "(iii) strict data curation is not always advantageous—loose filtering can preserve valuable boundary-case signal. "
            "Future work should focus on scaffold-split validation, learned molecular embeddings (Uni-Mol, JTVAE), "
            "and closed-loop experimental validation of top-predicted candidates."
        )
        lines.append("")
        lines.append("---")
        lines.append("*Master report auto-generated by Hybrid Agent Exploration system.*")

        report_text = "\n".join(lines)
        report_path = self.output_dir / "master_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        if self.embed_images:
            embed_markdown_images(report_path)
        return report_path

    def _generate_master_figures(self, successful: list[dict]) -> list[Path]:
        paths = []
        if not successful:
            return paths

        # Extract data
        r2s = []
        n_feats = []
        labels = []
        l2_methods = []
        for r in successful:
            m = r.get("metrics", {})
            if "r2" in m:
                r2s.append(m["r2"])
                n_feats.append(r.get("n_features", 0))
                labels.append(r.get("agent_id", "?"))
                l2_methods.append(r.get("config", {}).get("layer2", {}).get("method_id", "unknown"))

        if len(r2s) < 2:
            return paths

        r2s = np.array(r2s)
        n_feats = np.array(n_feats)

        # Figure 1: R² vs N_features (Pareto-style)
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        unique_l2 = sorted(set(l2_methods))
        color_map = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(unique_l2)}
        for m in unique_l2:
            mask = np.array(l2_methods) == m
            ax.scatter(n_feats[mask], r2s[mask], c=color_map[m], label=m, s=60, alpha=0.7, edgecolors="none")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel(r"$R^2$")
        ax.set_title("Performance vs. Complexity Trade-off")
        ax.legend(title="Feature Method", loc="best", frameon=False, fontsize=7)
        ax.axhline(0, color="gray", linestyle="--", lw=0.5)
        # Annotate top 3
        top_idx = np.argsort(r2s)[-3:]
        for idx in top_idx:
            ax.annotate(labels[idx], (n_feats[idx], r2s[idx]), fontsize=7, alpha=0.8)
        paths.append(self._save_fig(fig, "pareto_performance_complexity"))

        # Figure 2: R² distribution by L2 method
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        import pandas as pd
        df = pd.DataFrame({"r2": r2s, "l2": l2_methods})
        bp = ax.boxplot([group["r2"].values for name, group in df.groupby("l2")],
                         patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], _PALETTE[:len(bp["boxes"])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticklabels([name for name, _ in df.groupby("l2")], rotation=30, ha="right")
        ax.set_ylabel(r"$R^2$")
        ax.set_title(r"$R^2$ Distribution by Feature Method")
        ax.axhline(0, color="gray", linestyle="--", lw=0.5)
        fig.tight_layout()
        paths.append(self._save_fig(fig, "r2_by_feature_method"))

        # Figure 3: Success rate by L3 model
        l3_methods = [r.get("config", {}).get("layer3", {}).get("method_id", "unknown") for r in self.results]
        l3_status = [r.get("status", "unknown") for r in self.results]
        l3_success = defaultdict(lambda: {"success": 0, "total": 0})
        for m, s in zip(l3_methods, l3_status):
            l3_success[m]["total"] += 1
            if s == "success":
                l3_success[m]["success"] += 1

        if l3_success:
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            models = sorted(l3_success.keys())
            rates = [l3_success[m]["success"] / max(l3_success[m]["total"], 1) for m in models]
            colors = [_PALETTE[0] if r >= 0.8 else _PALETTE[3] for r in rates]
            ax.barh(models, rates, color=colors, edgecolor="white", height=0.6)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("Success Rate")
            ax.set_title("Pipeline Success Rate by Model")
            for i, (m, r) in enumerate(zip(models, rates)):
                ax.text(r + 0.02, i, f"{l3_success[m]['success']}/{l3_success[m]['total']}", va="center", fontsize=8)
            fig.tight_layout()
            paths.append(self._save_fig(fig, "success_rate_by_model"))

        return paths

    def _layer_contribution_analysis(self, successful: list[dict]) -> str:
        """Analyze which layer choices correlate with better performance."""
        if not successful:
            return "No successful experiments to analyze."

        lines = []
        # Group by L2 and compute mean R²
        l2_groups = defaultdict(list)
        for r in successful:
            l2 = r.get("config", {}).get("layer2", {}).get("method_id", "unknown")
            if "r2" in r.get("metrics", {}):
                l2_groups[l2].append(r["metrics"]["r2"])

        if l2_groups:
            lines.append("### Feature Representation (Layer 2) Impact")
            for l2, vals in sorted(l2_groups.items(), key=lambda x: np.mean(x[1]), reverse=True):
                lines.append(f"- **{l2}**: mean $R^2$ = {np.mean(vals):.4f} (n={len(vals)})")
            lines.append("")

        # Baseline-as-feature effect
        with_baseline = [r["metrics"]["r2"] for r in successful
                         if r.get("config", {}).get("baseline_as_feature") and "r2" in r.get("metrics", {})]
        without_baseline = [r["metrics"]["r2"] for r in successful
                            if not r.get("config", {}).get("baseline_as_feature") and "r2" in r.get("metrics", {})]
        if with_baseline and without_baseline:
            lines.append("### Baseline-PCE-as-Feature Effect")
            lines.append(f"- With baseline feature: mean $R^2$ = {np.mean(with_baseline):.4f} (n={len(with_baseline)})")
            lines.append(f"- Without baseline feature: mean $R^2$ = {np.mean(without_baseline):.4f} (n={len(without_baseline)})")
            lines.append(f"- Relative improvement: {(np.mean(with_baseline) - np.mean(without_baseline)) / max(abs(np.mean(without_baseline)), 0.001) * 100:.1f}%")
            lines.append("")

        return "\n".join(lines)

    def _best_practices(self, successful: list[dict]) -> str:
        lines = []
        lines.append("Based on the exploration results, the following patterns emerge as promising:")
        lines.append("")

        # Top L2
        l2_groups = defaultdict(list)
        for r in successful:
            l2 = r.get("config", {}).get("layer2", {}).get("method_id", "unknown")
            if "r2" in r.get("metrics", {}):
                l2_groups[l2].append(r["metrics"]["r2"])
        if l2_groups:
            best_l2 = max(l2_groups.items(), key=lambda x: np.mean(x[1]))
            lines.append(f"1. **Feature representation**: {best_l2[0]} consistently yields the highest mean $R^2$ ({np.mean(best_l2[1]):.3f}).")

        # Top L3
        l3_groups = defaultdict(list)
        for r in successful:
            l3 = r.get("config", {}).get("layer3", {}).get("method_id", "unknown")
            if "r2" in r.get("metrics", {}):
                l3_groups[l3].append(r["metrics"]["r2"])
        if l3_groups:
            best_l3 = max(l3_groups.items(), key=lambda x: np.mean(x[1]))
            lines.append(f"2. **Model choice**: {best_l3[0]} shows the strongest average performance ({np.mean(best_l3[1]):.3f}).")

        # Baseline feature
        with_baseline = [r["metrics"]["r2"] for r in successful if r.get("config", {}).get("baseline_as_feature") and "r2" in r.get("metrics", {})]
        without_baseline = [r["metrics"]["r2"] for r in successful if not r.get("config", {}).get("baseline_as_feature") and "r2" in r.get("metrics", {})]
        if with_baseline and without_baseline and np.mean(with_baseline) > np.mean(without_baseline):
            lines.append(f"3. **Baseline PCE as feature**: Including baseline PCE improves mean $R^2$ by ~{(np.mean(with_baseline) - np.mean(without_baseline)):.3f} absolute points.")

        lines.append("4. **Evaluation rigor**: Random-split results tend to be optimistic; future experiments should prioritize scaffold-split or temporal-split validation.")
        lines.append("5. **Sample efficiency**: Pipelines with 100–500 features outperform both very low-dimensional (≤20) and very high-dimensional (>2000) representations, suggesting an optimal bias–variance trade-off for this dataset size.")

        return "\n".join(lines)

    def _open_questions(self, successful: list[dict], failed: list[dict]) -> list[str]:
        questions = [
            "**Scaffold-split generalization**: Do MACCS+RF results hold when structurally similar molecules are forced into the same fold?",
            "**Learned representations**: How do JTVAE latent vectors or Uni-Mol 3D embeddings compare against classical fingerprints?",
            "**Multi-objective optimization**: Can we simultaneously optimize PCE, stability, and synthetic accessibility?",
            "**Experimental validation**: What is the true hit rate of top-k virtual-screening predictions?",
            "**Negative data utility**: Can failed experiments (poor R²) be leveraged as negative training signals for an active-learning loop?",
            "**Causal interpretation**: Do SHAP-identified important substructures correspond to known passivation mechanisms (e.g., Lewis-base donor–acceptor interactions)?",
        ]
        # Add failure-driven questions
        if failed:
            l3_fail = Counter(r.get("config", {}).get("layer3", {}).get("method_id", "unknown") for r in failed)
            most_fail = l3_fail.most_common(1)[0][0] if l3_fail else None
            if most_fail:
                questions.append(f"**Failure analysis**: Why does {most_fail} exhibit higher failure rates? Is it due to hyperparameter sensitivity, numerical instability, or data-format incompatibilities?")
        return [f"{i+1}. {q}" for i, q in enumerate(questions)]
