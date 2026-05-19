"""top_journal_report.py

Top-journal main-text report generator.
Produces a complete research article with 5-8 composite figures
and data-driven narrative.
"""

import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from .composite_figure import CompositeFigure, CompositeFigureTemplates
from .figure_selector import FigureSelector
from .image_embedder import embed_markdown_images
from .narrative_engine import NarrativeEngine
from .plan_registry import PlanRegistry, build_report_plan_context, load_plan_registry
from .report_bundle import ReportBundle
from .research_crew import ClaimAuditorAgent, ReviewerAgent
from .verified_discovery_ingestion import (
    format_verified_discovery_markdown,
    load_verified_discovery_summary,
)


class TopJournalReport:
    """Generate a top-journal-quality main-text report."""

    def __init__(self, results: list[dict], artifacts: dict[str, Any] | None = None,
                 output_dir: Path | str = "results/reports/top_journal",
                 quality_target: str = "standard",
                 embed_images: bool = False,
                 plan_registry: PlanRegistry | None = None,
                 plan_registry_path: Path | str | None = None,
                 enforce_plan_gates: bool = False):
        self.results = [r for r in results if isinstance(r, dict)]
        self.artifacts = artifacts or {}
        self.output_dir = Path(output_dir)
        self.quality_target = quality_target
        self.embed_images = embed_images
        if plan_registry is not None and plan_registry_path is not None:
            raise ValueError("Pass either plan_registry or plan_registry_path, not both")
        self.plan_registry = plan_registry or (
            load_plan_registry(plan_registry_path) if plan_registry_path else None
        )
        self.enforce_plan_gates = enforce_plan_gates
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.output_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.selector = FigureSelector(self.artifacts)
        self.narrative = NarrativeEngine()
        self.verified_discovery = load_verified_discovery_summary(self.artifacts)

    def generate(self) -> ReportBundle:
        """Generate the complete main-text report."""
        # 0. Prepare multi-model artifacts from results
        self._prepare_multi_model_artifacts()

        # 1. Determine which composite figures to generate
        main_composites = self.selector.select_composite_main(min_figures=5, max_figures=8)

        # 2. Generate composite figures
        figure_paths = []
        for comp in main_composites:
            path = self._build_composite_figure(comp)
            if path:
                figure_paths.append((comp.name, path))

        # 2b. Generate multi-model matrix figures if enough models
        matrix_figs = self._generate_matrix_figures()
        figure_paths.extend(matrix_figs)
        figure_paths.extend(self._generate_diagnostic_figures(len(figure_paths)))

        # 3. Assemble report
        best_result = self._get_best_result()
        best_r2 = best_result.get("metrics", {}).get("r2") if best_result else None
        claim_ledger = self._build_claim_ledger(figure_paths, best_result)
        report_text = self._assemble_report(figure_paths, best_result)
        auditor = ClaimAuditorAgent(max_supported_r2=best_r2)
        report_text = auditor.sanitize_text(report_text)

        report_path = self.output_dir / "main_text_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        if self.embed_images:
            embed_markdown_images(report_path)

        claim_ledger_path = self.output_dir / "claim_ledger.json"
        self._write_json(claim_ledger_path, claim_ledger)

        reviewer = ReviewerAgent()
        review = reviewer.review(report_text, [p for _, p in figure_paths], claim_ledger)
        audit = auditor.audit_text(report_text, claim_ledger)
        review_path = self.output_dir / "review_report.json"
        self._write_json(review_path, {"review": review, "claim_audit": audit})

        plan_manifest = None
        if self.plan_registry is not None:
            context = build_report_plan_context(
                results=self.results,
                artifacts=self.artifacts,
                figures=[p for _, p in figure_paths],
                report_text=report_text,
                claim_ledger=claim_ledger,
                review=review,
                audit=audit,
            )
            if self.enforce_plan_gates:
                evaluations = self.plan_registry.require_passed(context)
            else:
                evaluations = self.plan_registry.evaluate(context)
            plan_manifest = self.plan_registry.to_manifest(evaluations)

        manifest_path = self.output_dir / "run_manifest.json"
        manifest = self._build_manifest(
            figure_paths,
            best_result,
            claim_ledger_path,
            review_path,
            plan_manifest=plan_manifest,
        )
        self._write_json(manifest_path, manifest)

        warnings_out = list(review.get("findings", []))
        warnings_out.extend(item["phrase"] for item in audit.get("unsupported_claims", []))
        if plan_manifest is not None:
            warnings_out.extend(
                item["id"] for item in plan_manifest["plans"]
                if item["status"] != "passed"
            )
        quality_score = self._quality_score(len(figure_paths), review, audit)
        return ReportBundle(
            path=report_path,
            figures=[p for _, p in figure_paths],
            claim_ledger=claim_ledger,
            warnings=warnings_out,
            quality_score=quality_score,
            manifest_path=manifest_path,
            review_path=review_path,
        )

    def _prepare_multi_model_artifacts(self):
        """Extract per-model data from results into artifacts for matrix visualization.

        Only populates if not already present (load_artifacts may have done it).
        """
        if "multi_model_results" in self.artifacts and self.artifacts["multi_model_results"]:
            return
        multi_model = []
        for r in self.results:
            if r.get("status") != "success":
                continue
            cfg = r.get("config", {})
            m = r.get("metrics", {})
            artifacts = r.get("metrics", {}).get("_artifacts", {})
            model_entry = {
                "name": f"{cfg.get('layer2', {}).get('method_id', '?')}&{cfg.get('layer3', {}).get('method_id', '?')}",
                "l2": cfg.get("layer2", {}).get("method_id", "?"),
                "l3": cfg.get("layer3", {}).get("method_id", "?"),
                "r2": m.get("r2"),
                "rmse": m.get("rmse"),
                "mae": m.get("mae"),
                "pearson_r": m.get("pearson_r"),
                "y_true": artifacts.get("y_true"),
                "y_pred": artifacts.get("y_pred"),
                "feature_importances": artifacts.get("feature_importances"),
                "shap_values": artifacts.get("shap_values"),
            }
            multi_model.append(model_entry)
        if multi_model:
            self.artifacts["multi_model_results"] = multi_model

    def _generate_matrix_figures(self) -> list[tuple[str, Path]]:
        """Generate multi-model matrix comparison figures (like JPCL Figure 2-3)."""
        from .figure_generator import FigureGenerator
        fg = FigureGenerator(self.fig_dir, "matrix")
        paths = []

        multi = self.artifacts.get("multi_model_results", [])
        if len(multi) < 2:
            return paths

        # Figure: Multi-model parity grid
        parity_results = [m for m in multi if m.get("y_true") and m.get("y_pred")]
        if parity_results:
            p = fg.multi_model_parity_grid(parity_results, ncols=min(4, len(parity_results)))
            if p:
                paths.append(("multi_model_parity_grid", p))

        # Figure: Performance comparison bars
        perf_results = [{"name": m["name"], "r2": m.get("r2"), "rmse": m.get("rmse"),
                         "pearson_r": m.get("pearson_r"), "mae": m.get("mae")} for m in multi]
        p = fg.performance_comparison_bars(perf_results)
        if p:
            paths.append(("performance_comparison", p))

        return paths

    def _generate_diagnostic_figures(self, current_count: int) -> list[tuple[str, Path]]:
        """Generate evidence-grade diagnostics from available run metrics."""
        paths: list[tuple[str, Path]] = []
        builders = [
            ("ranked_model_performance", self._plot_ranked_model_performance),
            ("split_protocol_comparison", self._plot_split_protocol_comparison),
            ("feature_model_grid", self._plot_feature_model_grid),
            ("metric_tradeoff", self._plot_metric_tradeoff),
            ("evidence_ladder", self._plot_evidence_ladder),
            ("workflow_map", self._plot_workflow_map),
            ("design_rule_map", self._plot_design_rule_map),
        ]
        for name, builder in builders:
            if current_count + len(paths) >= 8:
                break
            try:
                path = builder()
            except Exception:
                path = None
            if path:
                paths.append((name, path))
        return paths

    def _successful_results(self) -> list[dict]:
        return [
            r for r in self.results
            if r.get("status") == "success" and r.get("metrics", {}).get("r2") is not None
        ]

    def _model_label(self, result: dict) -> str:
        cfg = result.get("config", {})
        feature = cfg.get("layer2", {}).get("method_id", "features")
        model = cfg.get("layer3", {}).get("method_id", "model")
        split = cfg.get("layer4", {}).get("method_id", "split")
        return f"{feature} | {model} | {split}"

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        path = self.fig_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_ranked_model_performance(self) -> Path | None:
        rows = sorted(self._successful_results(), key=lambda r: r["metrics"]["r2"])
        if not rows:
            return None
        labels = [self._model_label(r) for r in rows]
        values = [float(r["metrics"]["r2"]) for r in rows]
        fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.36 * len(rows))))
        colors = ["#2ca02c" if value >= 0 else "#d62728" for value in values]
        ax.barh(range(len(rows)), values, color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(r"$R^2$")
        ax.set_title("Ranked model performance from completed runs")
        handles = [
            plt.Rectangle((0, 0), 1, 1, color="#2ca02c", label=r"$R^2 \geq 0$"),
            plt.Rectangle((0, 0), 1, 1, color="#d62728", label=r"$R^2 < 0$"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False)
        return self._save_figure(fig, "diagnostic_ranked_model_performance.png")

    def _plot_split_protocol_comparison(self) -> Path | None:
        rows = self._successful_results()
        if not rows:
            return None
        groups: dict[str, list[float]] = {}
        for row in rows:
            split = row.get("config", {}).get("layer4", {}).get("method_id", "unknown_split")
            groups.setdefault(split, []).append(float(row["metrics"]["r2"]))
        labels = sorted(groups)
        fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(labels)), 4.0))
        for idx, label in enumerate(labels):
            vals = groups[label]
            x = np.full(len(vals), idx, dtype=float)
            if len(vals) > 1:
                x += np.linspace(-0.08, 0.08, len(vals))
            color = plt.cm.tab10(idx % 10)
            ax.scatter(x, vals, s=55, alpha=0.85, color=color, label=label)
            ax.plot([idx - 0.2, idx + 0.2], [np.mean(vals), np.mean(vals)], color=color, linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(r"$R^2$")
        ax.set_title("Validation protocol comparison, including scaffold split runs")
        ax.legend(title="Validation protocol", loc="best", frameon=False, fontsize=8, title_fontsize=8)
        fig.tight_layout()
        return self._save_figure(fig, "diagnostic_split_protocol_comparison.png")

    def _plot_feature_model_grid(self) -> Path | None:
        rows = self._successful_results()
        if not rows:
            return None
        features = sorted({r.get("config", {}).get("layer2", {}).get("method_id", "features") for r in rows})
        models = sorted({r.get("config", {}).get("layer3", {}).get("method_id", "model") for r in rows})
        if not features or not models:
            return None
        matrix = np.full((len(features), len(models)), np.nan)
        for i, feature in enumerate(features):
            for j, model in enumerate(models):
                vals = [
                    float(r["metrics"]["r2"]) for r in rows
                    if r.get("config", {}).get("layer2", {}).get("method_id", "features") == feature
                    and r.get("config", {}).get("layer3", {}).get("method_id", "model") == model
                ]
                if vals:
                    matrix[i, j] = float(np.mean(vals))
        fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * len(models)), max(3.6, 0.6 * len(features))))
        im = ax.imshow(np.nan_to_num(matrix, nan=0.0), cmap="RdYlGn", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.75, label=r"mean $R^2$")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        for i in range(len(features)):
            for j in range(len(models)):
                label = "NA" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8)
        ax.set_title("Feature representation by model family")
        fig.tight_layout()
        return self._save_figure(fig, "diagnostic_feature_model_grid.png")

    def _plot_metric_tradeoff(self) -> Path | None:
        rows = [r for r in self._successful_results() if r.get("metrics", {}).get("rmse") is not None]
        if not rows:
            return None
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        for row in rows:
            metrics = row["metrics"]
            ax.scatter(float(metrics["rmse"]), float(metrics["r2"]), s=65, alpha=0.8)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("RMSE")
        ax.set_ylabel(r"$R^2$")
        ax.set_title("Error-performance tradeoff across completed runs")
        return self._save_figure(fig, "diagnostic_metric_tradeoff.png")

    def _plot_evidence_ladder(self) -> Path | None:
        rows = self._successful_results()
        if not rows:
            return None
        splits = [r.get("config", {}).get("layer4", {}).get("method_id", "") for r in rows]
        evidence = {
            "successful runs": len(rows),
            "random split runs": sum("random" in split.lower() for split in splits),
            "scaffold split runs": sum("scaffold" in split.lower() for split in splits),
            "prediction arrays": int(bool(self.artifacts.get("y_true") and self.artifacts.get("y_pred"))),
            "SHAP diagnostics": int(bool(self.artifacts.get("shap_values"))),
            "external validation": int(bool(self.artifacts.get("y_true_external") and self.artifacts.get("y_pred_external"))),
        }
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        labels = list(evidence)
        values = list(evidence.values())
        colors = ["#1f77b4" if value else "#bdbdbd" for value in values]
        ax.bar(range(len(labels)), values, color=colors, edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Available evidence count")
        ax.set_title("Evidence-grade ladder and explicit validation gaps")
        fig.tight_layout()
        return self._save_figure(fig, "diagnostic_evidence_ladder.png")

    def _plot_workflow_map(self) -> Path | None:
        rows = self._successful_results()
        if not rows:
            return None
        layers = ["layer1", "layer2", "layer3", "layer4", "layer5"]
        fig, ax = plt.subplots(figsize=(8.0, 3.2))
        ax.axis("off")
        for idx, layer in enumerate(layers):
            methods = {
                r.get("config", {}).get(layer, {}).get("method_id", "unknown")
                for r in rows
            }
            x = 0.08 + idx * 0.18
            ax.text(
                x, 0.55,
                f"{layer}\n{len(methods)} method(s)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f2f2f2", "edgecolor": "#666666"},
            )
            if idx < len(layers) - 1:
                ax.annotate(
                    "",
                    xy=(x + 0.10, 0.55),
                    xytext=(x + 0.06, 0.55),
                    xycoords=ax.transAxes,
                    arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.2},
                )
        ax.set_title("Observed cross-layer exploration workflow", y=0.92)
        return self._save_figure(fig, "diagnostic_workflow_map.png")

    def _plot_design_rule_map(self) -> Path | None:
        rows = self._successful_results()
        if not rows:
            return None
        feature_scores: dict[str, list[float]] = {}
        model_scores: dict[str, list[float]] = {}
        for row in rows:
            r2 = float(row["metrics"]["r2"])
            feature = row.get("config", {}).get("layer2", {}).get("method_id", "features")
            model = row.get("config", {}).get("layer3", {}).get("method_id", "model")
            feature_scores.setdefault(feature, []).append(r2)
            model_scores.setdefault(model, []).append(r2)
        labels = [f"feature:{k}" for k in sorted(feature_scores)] + [f"model:{k}" for k in sorted(model_scores)]
        values = [np.mean(feature_scores[k]) for k in sorted(feature_scores)] + [
            np.mean(model_scores[k]) for k in sorted(model_scores)
        ]
        fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(labels)), 4.0))
        ax.bar(range(len(labels)), values, color="#1f77b4", alpha=0.85)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(r"mean $R^2$")
        ax.set_title("Run-derived design rule map")
        fig.tight_layout()
        return self._save_figure(fig, "diagnostic_design_rule_map.png")

    def _get_best_result(self) -> dict | None:
        successful = [r for r in self.results if r.get("status") == "success" and "r2" in r.get("metrics", {})]
        if not successful:
            return None
        return max(successful, key=lambda r: r["metrics"]["r2"])

    def _build_composite_figure(self, comp) -> Path | None:
        """Build a single composite figure from a CompositeFigureRequest.

        Uses composite_figure internal helpers that accept ax as first argument.
        """
        from .composite_figure import CompositeFigure
        from . import composite_figure as cf_helpers

        cf = CompositeFigure(layout=comp.layout,
                             figsize=(comp.layout[1] * 4.5 + 0.5, comp.layout[0] * 4.0 + 0.5))

        def _placeholder_on_ax(ax, text="Analysis not available"):
            ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, style="italic", color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

        def _coerce_shap_arrays(shap_values=None, shap_background=None, feature_names=None, **kw):
            if shap_values is None or shap_background is None:
                return None
            sv = np.asarray(shap_values, dtype=float)
            bg = np.asarray(shap_background, dtype=float)
            if sv.ndim == 1:
                sv = sv.reshape(-1, 1)
            if bg.ndim == 1:
                bg = bg.reshape(-1, 1)
            n_rows = min(sv.shape[0], bg.shape[0])
            n_cols = min(sv.shape[1], bg.shape[1])
            if n_rows < 2 or n_cols < 1:
                return None
            sv = np.nan_to_num(sv[:n_rows, :n_cols])
            bg = np.nan_to_num(bg[:n_rows, :n_cols])
            if feature_names and len(feature_names) >= n_cols:
                names = list(feature_names)[:n_cols]
            else:
                names = [f"Feature {i}" for i in range(n_cols)]
            return sv, bg, names

        def _top_shap_indices(sv: np.ndarray, top_k: int = 12) -> np.ndarray:
            mean_abs = np.mean(np.abs(sv), axis=0)
            return np.argsort(mean_abs)[::-1][:min(top_k, sv.shape[1])]

        def _plot_shap_bar_on_ax(ax, **kw):
            arrays = _coerce_shap_arrays(**kw)
            if arrays is None:
                ax.set_visible(False)
                return
            sv, _, names = arrays
            idx = _top_shap_indices(sv)[::-1]
            values = np.mean(np.abs(sv[:, idx]), axis=0)
            ax.barh(range(len(idx)), values, color="#1f77b4", edgecolor="white")
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels([names[i] for i in idx], fontsize=7)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Feature importance")

        def _plot_shap_swarm_on_ax(ax, **kw):
            arrays = _coerce_shap_arrays(**kw)
            if arrays is None:
                ax.set_visible(False)
                return
            sv, bg, names = arrays
            idx = _top_shap_indices(sv, top_k=10)
            rng = np.random.default_rng(42)
            scatter = None
            for rank, feature_idx in enumerate(idx[::-1]):
                y = np.full(sv.shape[0], rank, dtype=float)
                y += rng.normal(0.0, 0.055, size=sv.shape[0])
                scatter = ax.scatter(
                    sv[:, feature_idx], y, c=bg[:, feature_idx], cmap="viridis",
                    s=12, alpha=0.65, edgecolors="none"
                )
            ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels([names[i] for i in idx[::-1]], fontsize=7)
            ax.set_xlabel("SHAP value")
            ax.set_title("SHAP value distribution")
            if scatter is not None:
                cbar = ax.figure.colorbar(scatter, ax=ax, shrink=0.65, pad=0.02)
                cbar.set_label("Feature value", fontsize=7)
                cbar.ax.tick_params(labelsize=6)

        def _plot_shap_dependence_on_ax(ax, **kw):
            arrays = _coerce_shap_arrays(**kw)
            if arrays is None:
                ax.set_visible(False)
                return
            sv, bg, names = arrays
            idx = _top_shap_indices(sv, top_k=2)
            feature_idx = int(idx[0])
            color_idx = int(idx[1]) if len(idx) > 1 else feature_idx
            sc = ax.scatter(
                bg[:, feature_idx], sv[:, feature_idx], c=bg[:, color_idx],
                cmap="viridis", s=20, alpha=0.75, edgecolors="none"
            )
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_xlabel(names[feature_idx], fontsize=8)
            ax.set_ylabel("SHAP value", fontsize=8)
            ax.set_title(f"Dependence: {names[feature_idx]}", fontsize=9)
            cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.65, pad=0.02)
            cbar.set_label(names[color_idx], fontsize=7)
            cbar.ax.tick_params(labelsize=6)

        def _plot_shap_interaction_on_ax(ax, **kw):
            arrays = _coerce_shap_arrays(**kw)
            if arrays is None:
                ax.set_visible(False)
                return
            sv, bg, names = arrays
            idx = _top_shap_indices(sv, top_k=2)
            feature_i = int(idx[0])
            feature_j = int(idx[1]) if len(idx) > 1 else feature_i
            xi = bg[:, feature_i]
            yi = bg[:, feature_j]
            zi = np.abs(sv[:, feature_i])
            if np.ptp(xi) == 0 or np.ptp(yi) == 0:
                sc = ax.scatter(xi, yi, c=zi, cmap="magma", s=24, alpha=0.8, edgecolors="none")
            else:
                bins = 16
                xi_bins = np.linspace(float(xi.min()), float(xi.max()), bins + 1)
                yi_bins = np.linspace(float(yi.min()), float(yi.max()), bins + 1)
                heatmap = np.full((bins, bins), np.nan)
                counts = np.zeros((bins, bins), dtype=float)
                for x_val, y_val, z_val in zip(xi, yi, zi):
                    x_bin = min(np.digitize(x_val, xi_bins) - 1, bins - 1)
                    y_bin = min(np.digitize(y_val, yi_bins) - 1, bins - 1)
                    if np.isnan(heatmap[y_bin, x_bin]):
                        heatmap[y_bin, x_bin] = 0.0
                    heatmap[y_bin, x_bin] += z_val
                    counts[y_bin, x_bin] += 1
                with np.errstate(divide="ignore", invalid="ignore"):
                    heatmap = np.divide(heatmap, counts)
                heatmap = np.nan_to_num(heatmap)
                sc = ax.imshow(
                    heatmap, origin="lower", aspect="auto", cmap="magma",
                    extent=[xi_bins[0], xi_bins[-1], yi_bins[0], yi_bins[-1]]
                )
            ax.set_xlabel(names[feature_i], fontsize=8)
            ax.set_ylabel(names[feature_j], fontsize=8)
            ax.set_title("SHAP interaction map", fontsize=9)
            cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.65, pad=0.02)
            cbar.set_label("|SHAP|", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

        # Map atomic figure func_names to on_ax helpers in composite_figure module
        _AX_HELPERS = {
            "parity_plot": cf_helpers._plot_parity_on_ax,
            "residual_scatter": cf_helpers._plot_residual_scatter_on_ax,
            "residual_histogram": cf_helpers._plot_residual_hist_on_ax,
            "residual_qq": cf_helpers._plot_qq_on_ax,
            "data_distribution": cf_helpers._plot_distribution_on_ax,
            "feature_importance": cf_helpers._plot_feature_importance_on_ax,
            "cv_fold_performance": cf_helpers._plot_cv_folds_on_ax,
            # Method comparison fallbacks
            "pareto_front": cf_helpers._plot_pareto_on_ax,
            "radar_chart": cf_helpers._plot_radar_on_ax,
            "performance_comparison_bars": cf_helpers._plot_metric_comparison_bars,
            "bootstrap_confidence": lambda ax, **kw: _placeholder_on_ax(ax, "Bootstrap confidence interval"),
            "shap_bar": _plot_shap_bar_on_ax,
            "shap_swarm": _plot_shap_swarm_on_ax,
            "shap_dependence": _plot_shap_dependence_on_ax,
            "shap_interaction": _plot_shap_interaction_on_ax,
        }

        for sub in comp.subplots:
            if not self.selector._has_keys(sub.required_keys):
                continue
            plot_func = _AX_HELPERS.get(sub.func_name)
            if plot_func is None:
                continue

            # Build positional args and kwargs for the helper
            args = []
            kwargs = dict(sub.kwargs)
            for key in sub.required_keys:
                if key in self.artifacts:
                    val = self.artifacts[key]
                    # Map artifact keys to helper parameter names
                    if key == "y_true" and "y_pred" in sub.required_keys:
                        if not args:
                            args.append(val)  # y_true as first arg
                    elif key == "y_pred":
                        args.append(val)  # y_pred as second arg
                    elif key == "feature_importances":
                        args.append(val)
                    elif key == "cv_scores_per_fold":
                        args.append(val)
                    elif key == "y_values":
                        args.append(val)
                    else:
                        kwargs[key] = val

            # Special: parity plot needs r2
            if sub.func_name == "parity_plot":
                best = self._get_best_result()
                r2 = best.get("metrics", {}).get("r2") if best else None
                kwargs["r2"] = r2

            # Special: data distribution title
            if sub.func_name == "data_distribution":
                kwargs.setdefault("title", "Distribution")

            try:
                cf.add_subplot(plot_func, *args, **kwargs)
            except Exception:
                pass

        cf.add_overall_title(comp.title)
        # If no subplots were successfully added, skip this composite
        if cf._current_idx == 0:
            plt.close(cf.fig)
            return None
        return cf.save(self.fig_dir, f"fig_{comp.name}")

    def _assemble_report(self, figure_paths: list[tuple[str, Path]], best_result: dict | None) -> str:
        best_metrics = best_result.get("metrics", {}) if best_result else {}
        best_cfg = best_result.get("config", {}) if best_result else {}
        successful_count = len([r for r in self.results if r.get("status") == "success"])
        lines = [
            "# Machine-Learning-Accelerated Design of Molecular Additives for Perovskite Solar Cells",
            "",
            "## Abstract",
            self._abstract(best_metrics, best_cfg, successful_count),
            "",
            "**Keywords**: perovskite solar cells, machine learning, molecular additives, structure-property relationship, validation diagnostics, interpretability",
            "",
            "## 1. Introduction",
            self._introduction(),
            "",
            "## 2. Methods",
            self._methods(),
            "",
            "## 3. Results and Discussion",
            self._results_overview(best_result),
            "",
        ]

        for index, (name, path) in enumerate(figure_paths, 1):
            caption = self._figure_caption(name)
            rel = Path(path).name
            lines.append(self._figure_discussion(index, name, best_result))
            lines.append("")
            lines.append(f"**Figure {index}**. {caption}")
            lines.append("")
            lines.append(f"![Figure {index}: {name}](figures/{rel})")
            lines.append("")

        if self.verified_discovery:
            lines.extend([
                "### Verified Discovery Provenance",
                format_verified_discovery_markdown(self.verified_discovery),
                "",
            ])

        lines.extend([
            "## 4. Limitations and Evidence Gaps",
            self._limitations(),
            "",
            self._conclusion(best_result),
            "",
            "## References",
            self._reference_section(),
            "",
            "---",
            "*Report auto-generated by Hybrid Agent Exploration system with claim-audit sidecars.*",
        ])
        return "\n".join(lines)

    def _abstract(self, metrics: dict, config: dict, n_models: int) -> str:
        if self._has_training_only_metrics():
            return self._training_only_abstract(metrics, config, n_models)

        target = config.get("target", "delta_pce")
        r2 = metrics.get("r2")
        rmse = metrics.get("rmse")
        mae = metrics.get("mae")
        model = config.get("layer3", {}).get("method_id", "best observed model")
        feature = config.get("layer2", {}).get("method_id", "molecular representation")
        metric_sentence = "No successful metric summary is available."
        if r2 is not None:
            metric_sentence = f"The best observed configuration ({feature} with {model}) reached $R^2$ = {r2:.3f}"
            if rmse is not None:
                metric_sentence += f", RMSE = {rmse:.3f}"
            if mae is not None:
                metric_sentence += f", and MAE = {mae:.3f}"
            metric_sentence += "."
        return (
            f"We report an evidence-audited exploratory QSPR analysis for predicting {target} in perovskite solar-cell additive data. "
            f"The workflow evaluated {n_models} converged cross-layer configurations spanning data curation, molecular representation, model family, and validation protocol. "
            f"{metric_sentence} Random-split and scaffold-split results are reported separately so that apparent performance and generalization risk are not conflated. "
            "The report embeds all generated figures in context and records each quantitative claim in a machine-readable claim ledger."
        )

    def _training_only_abstract(self, metrics: dict, config: dict, n_models: int) -> str:
        target = config.get("target", "delta_pce")
        model = config.get("layer3", {}).get("method_id", "model")
        r2 = metrics.get("r2")
        rmse = metrics.get("rmse")
        mae = metrics.get("mae")
        metric_sentence = "No training metric summary is available."
        if r2 is not None:
            metric_sentence = f"The {model} fit reached a training-only $R^2$ = {r2:.3f}"
            if rmse is not None:
                metric_sentence += f", RMSE = {rmse:.3f}"
            if mae is not None:
                metric_sentence += f", and MAE = {mae:.3f}"
            metric_sentence += "."
        verification_note = (
            "The source-columns mode is marked smoke-only because evidence was accepted from source table columns rather than rechecked through external DOI and molecule services. "
            if self._is_source_columns_smoke()
            else ""
        )
        return (
            f"We report an evidence-audited exploratory QSPR package for predicting {target} in perovskite solar-cell additive data. "
            f"The package contains {n_models} converged verified-discovery model fit, a verified candidate table, report sidecars, and provenance manifests. "
            f"{metric_sentence} These metrics are training-only fit diagnostics and are not presented as validation, external validation, or chemical generalization evidence. "
            f"{verification_note}"
            "The report records supported claims in a machine-readable claim ledger and explicitly separates artifact availability from scientific validation."
        )

    def _results_overview(self, best_result: dict | None) -> str:
        successful = self._successful_results()
        if not successful or not best_result:
            return "No successful experiments were available for the main results narrative."
        best = best_result["metrics"]
        cfg = best_result.get("config", {})
        if self._has_training_only_metrics():
            smoke_note = (
                "The source-columns evidence mode is smoke-only and must be replaced by external-cached verification before publication-grade claims are made. "
                if self._is_source_columns_smoke()
                else ""
            )
            return (
                f"The verified-discovery package contains {len(successful)} successful training-only model fit. "
                f"The run used {cfg.get('layer2', {}).get('method_id', 'unknown')} features and "
                f"{cfg.get('layer3', {}).get('method_id', 'unknown')} with training-only $R^2$ = {best.get('r2', 0):.3f}. "
                "This value is a fit diagnostic from the verified training rows, not a random-split, scaffold-split, temporal-split, or external-validation result. "
                f"{smoke_note}"
                "Candidate ranking is therefore interpreted as a reproducibility and provenance check rather than a validated design rule."
            )
        split_counts = {}
        for row in successful:
            split = row.get("config", {}).get("layer4", {}).get("method_id", "unknown")
            split_counts[split] = split_counts.get(split, 0) + 1
        return (
            f"The completed experiment table contains {len(successful)} successful model runs. "
            f"The best observed run used {cfg.get('layer2', {}).get('method_id', 'unknown')} features and "
            f"{cfg.get('layer3', {}).get('method_id', 'unknown')} with $R^2$ = {best.get('r2', 0):.3f}. "
            f"Validation protocols represented in the run table include {', '.join(f'{k} (n={v})' for k, v in sorted(split_counts.items()))}. "
            "The scaffold split entries are treated as generalization diagnostics rather than external validation."
        )

    def _figure_discussion(self, index: int, name: str, best_result: dict | None) -> str:
        best_r2 = best_result.get("metrics", {}).get("r2") if best_result else None
        if name == "model_performance":
            return f"Figure {index} summarizes parity and residual diagnostics for the available prediction arrays; the best supported $R^2$ is {best_r2:.3f}." if best_r2 is not None else f"Figure {index} summarizes parity and residual diagnostics for the available prediction arrays."
        if name == "interpretability":
            return f"Figure {index} reports interpretability diagnostics from the supplied SHAP artifacts; these plots are descriptive and do not establish causal molecular mechanisms."
        if name in {"method_comparison", "performance_comparison", "multi_model_parity_grid"}:
            return f"Figure {index} compares completed model configurations so model-family and representation effects can be inspected from the same run table."
        if name == "split_protocol_comparison":
            return f"Figure {index} separates random and scaffold split performance to expose the validation gap relevant to chemical generalization."
        if name == "evidence_ladder":
            return f"Figure {index} is an evidence-grade map: bars with zero count indicate explicit gaps, including absent independent external validation when no external artifacts are present."
        if name == "workflow_map":
            return f"Figure {index} diagrams the observed cross-layer workflow using only method counts present in the successful run configurations."
        if name == "design_rule_map":
            return f"Figure {index} aggregates run-level $R^2$ values into a design-rule map, which should be read as exploratory ranking rather than a validated screening rule."
        return f"Figure {index} provides a run-derived diagnostic for {name.replace('_', ' ')} and is interpreted within the limits of the available artifacts."

    def _figure_caption(self, figure_name: str) -> str:
        captions = {
            "ranked_model_performance": "Ranked completed model runs by $R^2$; negative values indicate performance below the mean baseline.",
            "split_protocol_comparison": "Distribution of $R^2$ by validation protocol, including scaffold split diagnostics.",
            "feature_model_grid": "Mean $R^2$ for each observed feature-representation and model-family pair.",
            "metric_tradeoff": "RMSE versus $R^2$ for completed runs, showing the error-performance tradeoff.",
            "evidence_ladder": "Evidence-grade ladder derived from available artifacts; gray bars mark missing evidence classes.",
            "workflow_map": "Observed cross-layer exploration workflow summarized by method-count coverage per layer.",
            "design_rule_map": "Exploratory design-rule map from mean run-level $R^2$ grouped by representation and model family.",
        }
        return captions.get(figure_name, self.narrative.figure_caption(figure_name, {}))

    def _limitations(self) -> str:
        if self._has_training_only_metrics():
            source_note = (
                "Source-columns mode uses DOI and molecule fields already present in the input table and is explicitly smoke-only. "
                if self._is_source_columns_smoke()
                else "External-cached evidence checks are recorded separately from model validation. "
            )
            return (
                f"{source_note}"
                "The reported model metrics are training-only diagnostics and cannot establish predictive generalization. "
                "No scaffold-split, temporal-split, independent external-validation, or experimental follow-up evidence is inferred unless those artifacts are supplied. "
                "Mechanistic statements are limited to artifact availability and aggregate diagnostics; candidate rankings require external validation before they can be used as design rules."
            )
        has_external = bool(self.artifacts.get("y_true_external") and self.artifacts.get("y_pred_external"))
        external_note = (
            "Independent external validation artifacts were available and are reported as diagnostics."
            if has_external
            else "No independent external validation artifacts were supplied, so the report does not claim external validation."
        )
        return (
            "The analysis is bounded by the size and composition of the available experiment table. "
            "Random splits can overestimate generalization when structural analogues occur across train and test partitions, making scaffold split diagnostics essential. "
            f"{external_note} Mechanistic statements are limited to associations supported by model diagnostics and should be tested experimentally before use as design rules."
        )

    def _conclusion(self, best_result: dict | None) -> str:
        if not self._has_training_only_metrics():
            return self.narrative.conclusion(self.results)

        best_metrics = best_result.get("metrics", {}) if best_result else {}
        r2 = best_metrics.get("r2")
        if r2 is None:
            metric_sentence = "No successful training metric was available."
        else:
            metric_sentence = f"The current package records a training-only $R^2$ = {r2:.3f}, which is treated only as a fit diagnostic."
        smoke_sentence = (
            "Because this run used source-columns mode, it is a smoke package rather than a publication-grade verification package. "
            if self._is_source_columns_smoke()
            else ""
        )
        return "\n".join(
            [
                "## Conclusion",
                "",
                "This research package demonstrates that the verified-discovery artifact chain can produce a traceable dataset, candidate library, ranked candidates, manuscript text, SI, and provenance manifests from real input records.",
                f"{metric_sentence} {smoke_sentence}The package does not support claims about representation superiority, baseline-feature benefit, SHAP-derived mechanisms, scaffold generalization, or externally validated candidate performance unless those artifacts are added in a future run.",
                "The next scientific step is to replace smoke evidence with external-cached DOI and molecule verification, add held-out validation protocols, and connect top candidates to independently verifiable experimental or supplier evidence.",
            ]
        )

    def _build_claim_ledger(self, figure_paths: list[tuple[str, Path]], best_result: dict | None) -> list[dict[str, Any]]:
        ledger: list[dict[str, Any]] = []
        successful = self._successful_results()
        best_metrics = best_result.get("metrics", {}) if best_result else {}
        best_cfg = best_result.get("config", {}) if best_result else {}
        ledger.append({
            "claim": "Best observed model coefficient of determination",
            "evidence_id": "metric:best_model.r2",
            "value": best_metrics.get("r2"),
            "source": "results.metrics.r2",
        })
        if best_metrics.get("rmse") is not None:
            ledger.append({
                "claim": "Best observed model root mean squared error",
                "evidence_id": "metric:best_model.rmse",
                "value": best_metrics.get("rmse"),
                "source": "results.metrics.rmse",
            })
        ledger.append({
            "claim": "Number of successful model runs",
            "evidence_id": "metric:successful_runs.count",
            "value": len(successful),
            "source": "results.status",
        })
        scaffold_count = sum(
            "scaffold" in r.get("config", {}).get("layer4", {}).get("method_id", "").lower()
            for r in successful
        )
        ledger.append({
            "claim": "Scaffold split diagnostics are available in the run table",
            "evidence_id": "metric:evaluation.scaffold_count",
            "value": scaffold_count,
            "source": "results.config.layer4.method_id",
        })
        if best_cfg:
            ledger.append({
                "claim": "Best observed feature and model configuration",
                "evidence_id": "metric:best_model.config",
                "value": {
                    "feature": best_cfg.get("layer2", {}).get("method_id"),
                    "model": best_cfg.get("layer3", {}).get("method_id"),
                    "split": best_cfg.get("layer4", {}).get("method_id"),
                },
                "source": "results.config",
            })
        for index, (name, path) in enumerate(figure_paths, 1):
            ledger.append({
                "claim": f"Figure {index} supports the {name.replace('_', ' ')} discussion",
                "evidence_id": f"figure:fig{index}",
                "figure": str(Path("figures") / Path(path).name),
            })
        if self.verified_discovery:
            ledger.extend([
                {
                    "claim": "Number of top verified candidate records ingested into the report bundle",
                    "evidence_id": "discovery:top_candidates.count",
                    "value": len(self.verified_discovery.get("top_candidates", [])),
                    "source": "discovery/ranked_candidates.csv",
                },
                {
                    "claim": "Quarantine reason counts from verified discovery provenance",
                    "evidence_id": "provenance:quarantine_reason_summary",
                    "value": self.verified_discovery.get("quarantine_reason_summary", {}),
                    "source": "dataset/quarantine.csv",
                },
            ])
        evidence_context = self._evidence_context()
        if evidence_context:
            ledger.append({
                "claim": "Research package evidence context and smoke status",
                "evidence_id": "provenance:research_package.evidence_context",
                "value": evidence_context,
                "source": "artifacts.evidence_context",
            })
        references = self._load_references()
        ledger.append({
            "claim": "Literature benchmark covers PSC, molecular ML, virtual screening, experimental validation, and database subfields",
            "evidence_id": "literature:reference_library.count",
            "value": len(references),
            "source": "docs/references.bib",
        })
        return ledger

    def _build_manifest(
        self,
        figure_paths: list[tuple[str, Path]],
        best_result: dict | None,
        claim_ledger_path: Path,
        review_path: Path,
        plan_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        best_metrics = best_result.get("metrics", {}) if best_result else {}
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "quality_target": self.quality_target,
            "n_results": len(self.results),
            "n_successful": len([r for r in self.results if r.get("status") == "success"]),
            "best_model": {
                "r2": best_metrics.get("r2"),
                "rmse": best_metrics.get("rmse"),
                "mae": best_metrics.get("mae"),
                "pearson_r": best_metrics.get("pearson_r"),
            },
            "figures": [
                {"id": f"figure:fig{index}", "name": name, "path": str(Path("figures") / Path(path).name)}
                for index, (name, path) in enumerate(figure_paths, 1)
            ],
            "references": {
                "count": len(self._load_references()),
                "source": "docs/references.bib",
            },
            "claim_ledger_path": str(claim_ledger_path.name),
            "review_path": str(review_path.name),
            "agents": ["TopJournalReport", "FigureAssemblyAgent", "NarrativeEngine", "ReviewerAgent", "ClaimAuditorAgent"],
        }
        if plan_manifest is not None:
            manifest["plan_registry"] = plan_manifest
            manifest["agents"].insert(0, "PlanRegistryAgent")
        if self.verified_discovery:
            manifest["verified_discovery"] = self.verified_discovery
        evidence_context = self._evidence_context()
        if evidence_context:
            manifest["evidence_context"] = evidence_context
        return manifest

    def _quality_score(self, figure_count: int, review: dict[str, Any], audit: dict[str, Any]) -> float:
        score = 0.45 + min(figure_count, 8) * 0.05
        if review.get("passed"):
            score += 0.10
        if audit.get("passed"):
            score += 0.05
        if not review.get("passed") or not audit.get("passed"):
            score = min(score, 0.69)
        return round(min(score, 1.0), 3)

    def _write_json(self, path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=self._json_default)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    def _reference_section(self, max_references: int = 40) -> str:
        references = self._load_references(max_references=max_references)
        if not references:
            return "Reference library not available."
        lines = []
        for idx, ref in enumerate(references, 1):
            doi = ref.get("doi")
            doi_text = f" DOI: {doi}." if doi else ""
            lines.append(
                f"{idx}. {ref.get('author', 'Unknown authors')}. "
                f"{ref.get('title', 'Untitled')}. "
                f"{ref.get('journal', 'Unknown journal')} {ref.get('year', 'n.d.')}."
                f"{doi_text}"
            )
        return "\n".join(lines)

    def _load_references(self, max_references: int | None = None) -> list[dict[str, str]]:
        bib_path = Path(__file__).resolve().parents[2] / "docs" / "references.bib"
        if not bib_path.exists():
            return []
        text = bib_path.read_text(encoding="utf-8", errors="ignore")
        entries = [entry for entry in re.split(r"\n(?=@\w+\{)", text) if entry.strip().startswith("@")]
        references: list[dict[str, str]] = []
        for entry in entries:
            references.append({
                "title": self._extract_bib_field(entry, "title"),
                "author": self._extract_bib_field(entry, "author"),
                "year": self._extract_bib_field(entry, "year"),
                "journal": self._extract_bib_field(entry, "journal"),
                "doi": self._extract_bib_field(entry, "doi"),
            })
            if max_references is not None and len(references) >= max_references:
                break
        return references

    @staticmethod
    def _extract_bib_field(entry: str, field: str) -> str:
        match = re.search(rf"\b{field}\s*=\s*\{{(.*?)\}}\s*,", entry, flags=re.DOTALL)
        if not match:
            return ""
        return " ".join(match.group(1).split())

    def _has_shap_artifacts(self) -> bool:
        return bool(self.artifacts.get("shap_values") and self.artifacts.get("shap_background"))

    def _evidence_context(self) -> dict[str, Any]:
        value = self.artifacts.get("evidence_context")
        return value if isinstance(value, dict) else {}

    def _is_source_columns_smoke(self) -> bool:
        context = self._evidence_context()
        return bool(context.get("source_columns_is_smoke_only"))

    def _has_training_only_metrics(self) -> bool:
        context = self._evidence_context()
        if context.get("metric_scope") == "training_only":
            return True
        return any(
            row.get("metrics", {}).get("metric_scope") == "training_only"
            for row in self.results
        )

    def _introduction(self) -> str:
        if self._has_training_only_metrics():
            smoke_note = (
                "This source-columns package is a smoke run: DOI and molecule evidence were accepted from the input columns rather than rechecked through external services. "
                if self._is_source_columns_smoke()
                else "External evidence checks are recorded in the package manifests, but model metrics remain training-only. "
            )
            return (
                "Accurate prediction of power-conversion-efficiency (PCE) modulation by molecular additives "
                "requires both trustworthy source records and validation protocols that test chemical generalization. "
                "This package focuses on the reproducible artifact chain needed for that study: strict row verification, "
                "candidate-library normalization, training-set model fitting, candidate ranking, manuscript sidecars, and provenance manifests. "
                f"{smoke_note}"
                "The generated text therefore treats all model metrics as training-only diagnostics and reserves scientific design-rule claims for future runs with validation artifacts."
            )
        interpretation_sentence = (
            "Available SHAP diagnostics are used to describe model associations while avoiding causal mechanistic claims. "
            if self._has_shap_artifacts()
            else "Interpretability is limited to non-SHAP diagnostics because SHAP artifacts were not supplied for this run. "
        )
        return (
            "Accurate prediction of power-conversion-efficiency (PCE) modulation by molecular additives "
            "is a central challenge in perovskite solar-cell (PSC) materials design. "
            "Traditional Edisonian experimentation is too slow to explore the vast chemical space of "
            "potential passivators, hole-transport materials, and anti-solvent additives. "
            "Machine learning offers a data-driven route, yet its utility depends critically on the "
            "joint choices of data curation, molecular representation, model architecture, and evaluation protocol. "
            "Here we systematically explore this cross-layer design space via a multi-agent exploration framework, "
            "evaluating diverse combinations of cleaning strategies, molecular fingerprints, machine-learning algorithms, "
            "and validation protocols. "
            f"{interpretation_sentence}"
            "This work provides a reproducible, scalable platform for accelerating photovoltaic materials discovery."
        )

    def _methods(self) -> str:
        successful = [r for r in self.results if r.get("status") == "success"]
        if not successful:
            return "No successful experiments to describe."
        # Collect unique methods used
        l1s = set(r.get("config", {}).get("layer1", {}).get("method_id", "?") for r in successful)
        l2s = set(r.get("config", {}).get("layer2", {}).get("method_id", "?") for r in successful)
        l3s = set(r.get("config", {}).get("layer3", {}).get("method_id", "?") for r in successful)
        l4s = set(r.get("config", {}).get("layer4", {}).get("method_id", "?") for r in successful)

        if self._has_training_only_metrics():
            lines = [
                "### Data Collection and Curation",
                f"Data were curated with {', '.join(sorted(l1s))}. Rows lacking required DOI, molecule, or target fields were quarantined rather than used for model fitting.",
                "",
                "### Feature Engineering",
                f"The package used the run-declared molecular representation(s): {', '.join(sorted(l2s))}. No additional Morgan, SHAP, or baseline-feature comparisons are inferred unless corresponding artifacts are supplied.",
                "",
                "### Model Fitting and Candidate Ranking",
                f"Model class(es) recorded in the package: {', '.join(sorted(l3s))}. Metrics emitted by this runner are training-only diagnostics from verified rows. Candidate ranking uses the validated candidate-library contract and records score components in the discovery manifest.",
                "",
                "### Validation Status",
                "No cross-validation, scaffold-split, temporal-split, independent external-validation, or experimental follow-up metric is inferred from this runner output. External evidence and model validation must be added as explicit artifacts before publication-grade claims are made.",
            ]
            return "\n".join(lines)

        lines = [
            "### Data Collection and Curation",
            f"Data were curated using the following strategies: {', '.join(sorted(l1s))}. "
            "Low-quality entries with physically implausible PCE values were removed to ensure reliable structure–property modeling.",
            "",
            "### Feature Engineering",
            f"Molecular representations evaluated included: {', '.join(sorted(l2s))}. "
            "All features were standardized (zero mean, unit variance) prior to model training.",
            "",
            "### Model Training and Selection",
            f"Machine-learning algorithms tested: {', '.join(sorted(l3s))}. "
            "Hyperparameters were optimized via grid search or Optuna where applicable. "
            f"Evaluation protocols: {', '.join(sorted(l4s))}.",
            "",
            "### Interpretability and Validation",
        ]
        if self._has_shap_artifacts():
            lines.append(
                "SHAP (SHapley Additive exPlanations) diagnostics were generated from supplied artifacts to describe feature associations."
            )
        else:
            lines.append(
                "SHAP diagnostics were not available in the supplied artifacts; interpretability claims are therefore limited to aggregate feature/model diagnostics."
            )
        lines.append(
            "External validation is reported only when independent external artifacts are supplied; otherwise scaffold splits are treated as generalization diagnostics."
        )
        return "\n".join(lines)
