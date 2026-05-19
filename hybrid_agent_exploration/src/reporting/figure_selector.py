"""figure_selector.py

Intelligent figure selector — decide which atomic plots and composite figures
to generate based on data availability and scientific importance.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AtomicFigureRequest:
    """Request to generate a single atomic figure."""
    name: str
    func_name: str
    priority: int  # 1 = highest, 10 = lowest
    required_keys: list[str]  # keys that must exist in artifacts/metrics
    kwargs: dict = field(default_factory=dict)


@dataclass
class CompositeFigureRequest:
    """Request to generate a composite figure from multiple atomic plots."""
    name: str
    title: str
    layout: tuple[int, int]  # (nrows, ncols)
    subplots: list[AtomicFigureRequest]
    priority: int
    section: str  # "main" or "si"


# ---------------------------------------------------------------------------
# Registry of all possible figures
# ---------------------------------------------------------------------------

ATOMIC_REGISTRY: list[AtomicFigureRequest] = [
    AtomicFigureRequest("parity", "parity_plot", 1,
                        ["y_true", "y_pred"]),
    AtomicFigureRequest("residual_scatter", "residual_scatter", 2,
                        ["y_true", "y_pred"]),
    AtomicFigureRequest("residual_hist", "residual_histogram", 3,
                        ["y_true", "y_pred"]),
    AtomicFigureRequest("residual_qq", "residual_qq", 4,
                        ["y_true", "y_pred"]),
    AtomicFigureRequest("feature_importance", "feature_importance", 2,
                        ["feature_importances"]),
    AtomicFigureRequest("cv_folds", "cv_fold_performance", 3,
                        ["cv_scores_per_fold"]),
    AtomicFigureRequest("correlation_heatmap", "correlation_heatmap", 4,
                        ["X"]),
    AtomicFigureRequest("feature_space_umap", "feature_space_projection", 5,
                        ["X"], {"method": "umap"}),
    AtomicFigureRequest("feature_space_tsne", "feature_space_projection", 6,
                        ["X"], {"method": "tsne"}),
    AtomicFigureRequest("data_distribution", "data_distribution", 2,
                        ["y_values"]),
    AtomicFigureRequest("shap_bar", "shap_bar", 2,
                        ["shap_values", "shap_background"]),
    AtomicFigureRequest("shap_swarm", "shap_swarm", 2,
                        ["shap_values", "shap_background"]),
    AtomicFigureRequest("shap_dependence", "shap_dependence", 3,
                        ["shap_values", "shap_background", "feature_names"]),
    AtomicFigureRequest("shap_interaction", "shap_interaction", 4,
                        ["shap_values", "shap_background", "feature_names"]),
    AtomicFigureRequest("external_validation", "external_validation", 2,
                        ["y_true_external", "y_pred_external"]),
    AtomicFigureRequest("multi_model_grid", "multi_model_parity_grid", 2,
                        ["multi_model_results"]),
    AtomicFigureRequest("performance_bars", "performance_comparison_bars", 2,
                        ["multi_model_results"]),
]

COMPOSITE_REGISTRY: list[CompositeFigureRequest] = [
    CompositeFigureRequest(
        "data_overview", "Data Quality Overview", (1, 3),
        [
            AtomicFigureRequest("data_raw", "data_distribution", 1, ["y_raw"]),
            AtomicFigureRequest("data_cleaned", "data_distribution", 1, ["y_cleaned"]),
            AtomicFigureRequest("data_removed", "data_distribution", 1, ["y_removed"]),
        ],
        priority=1, section="main",
    ),
    CompositeFigureRequest(
        "model_performance", "Model Performance Overview", (2, 2),
        [
            AtomicFigureRequest("parity", "parity_plot", 1, ["y_true", "y_pred"]),
            AtomicFigureRequest("residual_scatter", "residual_scatter", 1, ["y_true", "y_pred"]),
            AtomicFigureRequest("residual_hist", "residual_histogram", 1, ["y_true", "y_pred"]),
            AtomicFigureRequest("residual_qq", "residual_qq", 1, ["y_true", "y_pred"]),
        ],
        priority=1, section="main",
    ),
    CompositeFigureRequest(
        "feature_space", "Feature Space and Representation", (1, 3),
        [
            AtomicFigureRequest("correlation", "correlation_heatmap", 1, ["X"]),
            AtomicFigureRequest("umap", "feature_space_projection", 1, ["X", "y_values"], {"method": "umap"}),
            AtomicFigureRequest("tsne", "feature_space_projection", 1, ["X", "y_values"], {"method": "tsne"}),
        ],
        priority=2, section="main",
    ),
    CompositeFigureRequest(
        "interpretability", "Model Interpretability", (2, 2),
        [
            AtomicFigureRequest("shap_bar", "shap_bar", 1, ["shap_values", "shap_background"]),
            AtomicFigureRequest("shap_swarm", "shap_swarm", 1, ["shap_values", "shap_background"]),
            AtomicFigureRequest("dependence", "shap_dependence", 1, ["shap_values", "shap_background"]),
            AtomicFigureRequest("interaction", "shap_interaction", 1, ["shap_values", "shap_background"]),
        ],
        priority=2, section="main",
    ),
    CompositeFigureRequest(
        "validation", "Validation and Generalization", (1, 3),
        [
            AtomicFigureRequest("cv_folds", "cv_fold_performance", 1, ["cv_scores_per_fold"]),
            AtomicFigureRequest("external", "external_validation", 1, ["y_true_external", "y_pred_external"]),
            AtomicFigureRequest("bootstrap", "bootstrap_confidence", 1, ["y_true", "y_pred"]),
        ],
        priority=3, section="main",
    ),
    CompositeFigureRequest(
        "method_comparison", "Method Comparison and Selection", (1, 3),
        [
            AtomicFigureRequest("pareto", "pareto_front", 1, ["multi_model_results"]),
            AtomicFigureRequest("radar", "radar_chart", 1, ["multi_model_results"]),
            AtomicFigureRequest("performance", "performance_comparison_bars", 1, ["multi_model_results"]),
        ],
        priority=3, section="main",
    ),
    CompositeFigureRequest(
        "molecular_discovery", "Molecular Discovery and Screening", (1, 3),
        [
            AtomicFigureRequest("top_k", "molecular_grid", 1, ["top_k_smiles"]),
            AtomicFigureRequest("similarity", "similarity_network", 1, ["similarity_matrix"]),
            AtomicFigureRequest("predicted", "predicted_vs_property", 1, ["predictions"]),
        ],
        priority=4, section="main",
    ),
    CompositeFigureRequest(
        "error_analysis", "Error Analysis and Limitations", (2, 2),
        [
            AtomicFigureRequest("error_bins", "error_by_feature_bins", 1, ["y_true", "y_pred", "feature_values"]),
            AtomicFigureRequest("outliers", "outlier_scatter", 1, ["y_true", "y_pred"]),
            AtomicFigureRequest("sensitivity", "sensitivity_analysis", 1, ["model", "X"]),
            AtomicFigureRequest("learning_curve", "learning_curve", 1, ["train_sizes", "train_scores", "test_scores"]),
        ],
        priority=4, section="main",
    ),
]


class FigureSelector:
    """Select which figures to generate based on available data."""

    def __init__(self, data: dict[str, Any]):
        """data: dict containing all available artifacts and metrics."""
        self.data = data

    def _has_keys(self, keys: list[str]) -> bool:
        """Check if all required keys exist and are non-empty."""
        for k in keys:
            if k not in self.data:
                return False
            v = self.data[k]
            if v is None:
                return False
            if isinstance(v, (list, np.ndarray)) and len(v) == 0:
                return False
        return True

    def select_atomic(self, max_figures: int = 50) -> list[AtomicFigureRequest]:
        """Select atomic figures that can be generated."""
        available = []
        for req in ATOMIC_REGISTRY:
            if self._has_keys(req.required_keys):
                available.append(req)
        # Sort by priority
        available.sort(key=lambda r: r.priority)
        return available[:max_figures]

    def select_composite_main(self, min_figures: int = 5, max_figures: int = 8) -> list[CompositeFigureRequest]:
        """Select composite figures for the main text (top-journal style).

        Strategy: rank composites by how many subplots are available,
        then pick top 5-8.
        """
        candidates = []
        for comp in COMPOSITE_REGISTRY:
            if comp.section != "main":
                continue
            available_subs = sum(1 for sub in comp.subplots if self._has_keys(sub.required_keys))
            if available_subs >= 2:  # At least 2 subplots to form a meaningful composite
                candidates.append((comp, available_subs))

        # Sort by available subplots desc, then by priority
        candidates.sort(key=lambda x: (-x[1], x[0].priority))
        selected = [c[0] for c in candidates[:max_figures]]
        if len(selected) < min_figures:
            # Pad with SI-level composites if needed
            pass  # TODO: optionally promote SI composites
        return selected

    def select_si(self, max_figures: int = 30) -> list[AtomicFigureRequest]:
        """Select figures for Supporting Information.

        SI gets all atomic figures that were NOT used in main composites,
        plus any extras.
        """
        main_composites = self.select_composite_main()
        used_atomic_names = set()
        for comp in main_composites:
            for sub in comp.subplots:
                if self._has_keys(sub.required_keys):
                    used_atomic_names.add(sub.name)

        si_figures = []
        for req in ATOMIC_REGISTRY:
            if self._has_keys(req.required_keys) and req.name not in used_atomic_names:
                si_figures.append(req)

        # Also add any per-model SHAP figures if multiple models exist
        if "multi_model_results" in self.data:
            models = self.data["multi_model_results"]
            for i, m in enumerate(models):
                if m.get("shap_values") is not None:
                    si_figures.append(AtomicFigureRequest(
                        f"shap_model_{i}", "shap_swarm", 5,
                        [], {"model_name": m.get("name", f"model_{i}")}
                    ))

        si_figures.sort(key=lambda r: r.priority)
        return si_figures[:max_figures]

    def generate_report_summary(self) -> str:
        """Generate a text summary of what figures will be produced."""
        main = self.select_composite_main()
        si = self.select_si()
        atomic = self.select_atomic()
        lines = [
            "# Figure Generation Plan",
            f"",
            f"## Main Text Composite Figures: {len(main)}",
        ]
        for comp in main:
            n_available = sum(1 for sub in comp.subplots if self._has_keys(sub.required_keys))
            lines.append(f"- **{comp.name}** ({comp.layout[0]}×{comp.layout[1]}): {n_available}/{len(comp.subplots)} subplots available")
        lines.append("")
        lines.append(f"## Supporting Information Figures: {len(si)}")
        for req in si[:15]:
            lines.append(f"- {req.name}")
        if len(si) > 15:
            lines.append(f"- ... and {len(si) - 15} more")
        lines.append("")
        lines.append(f"## Total Atomic Figures Available: {len(atomic)}")
        return "\n".join(lines)
