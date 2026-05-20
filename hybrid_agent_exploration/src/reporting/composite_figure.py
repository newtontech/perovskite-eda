"""composite_figure.py

Composite Figure Engine — assemble 2-4 atomic figures into a single
top-journal-quality composite figure with sub-labels (a), (b), (c)...
"""

import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# Conservative, colorblind-friendly palette
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


class CompositeFigure:
    """Manage a composite figure with multiple subplots.

    Usage:
        cf = CompositeFigure(layout=(2, 2), figsize=(10, 8))
        cf.add_subplot(0, plot_parity, y_true, y_pred)
        cf.add_subplot(1, plot_residual, y_true, y_pred)
        cf.add_subplot(2, plot_hist, residuals)
        cf.add_subplot(3, plot_qq, residuals)
        cf.add_overall_title("Model Performance")
        path = cf.save(output_dir, "fig02_performance")
    """

    SUB_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    def __init__(self, layout: tuple[int, int] = (2, 2),
                 figsize: tuple[float, float] | None = None):
        self.nrows, self.ncols = layout
        self.n_subplots = self.nrows * self.ncols
        if figsize is None:
            figsize = (self.ncols * 4.5 + 0.5, self.nrows * 4.0 + 0.5)
        self.fig = plt.figure(figsize=figsize)
        self.gs = GridSpec(self.nrows, self.ncols, figure=self.fig,
                           wspace=0.35, hspace=0.40)
        self._current_idx = 0
        self._subplot_axes = []

    def add_subplot(self, plot_func: Callable, *args, title: str = "",
                    **kwargs) -> bool:
        """Add a subplot by calling plot_func(ax, *args, **kwargs).

        plot_func must accept an Axes object as its first argument.
        Returns True on success, False on failure (plot skipped).
        """
        if self._current_idx >= self.n_subplots:
            return False
        row = self._current_idx // self.ncols
        col = self._current_idx % self.ncols
        ax = self.fig.add_subplot(self.gs[row, col])
        try:
            plot_func(ax, *args, **kwargs)
            has_data_artists = bool(
                ax.lines or ax.collections or ax.patches or ax.images or ax.containers
            )
            if not ax.get_visible() or not has_data_artists:
                self.fig.delaxes(ax)
                return False
            label = self.SUB_LABELS[self._current_idx] if self._current_idx < len(self.SUB_LABELS) else ""
            if label:
                ax.text(-0.18, 1.05, label, transform=ax.transAxes,
                        fontsize=12, fontweight="bold", va="top", ha="right")
            if title:
                ax.set_title(title)
            self._subplot_axes.append(ax)
            self._current_idx += 1
            return True
        except Exception:
            self.fig.delaxes(ax)
            return False

    def get_next_ax(self) -> plt.Axes | None:
        """Get the next available Axes for manual plotting."""
        if self._current_idx >= self.n_subplots:
            return None
        row = self._current_idx // self.ncols
        col = self._current_idx % self.ncols
        ax = self.fig.add_subplot(self.gs[row, col])
        label = self.SUB_LABELS[self._current_idx] if self._current_idx < len(self.SUB_LABELS) else ""
        if label:
            ax.text(-0.18, 1.05, label, transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="top", ha="right")
        self._subplot_axes.append(ax)
        self._current_idx += 1
        return ax

    def add_overall_title(self, title: str):
        """Add a title for the entire composite figure."""
        self.fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    def save(self, output_dir: Path, suffix: str, fmt: str = "png") -> Path:
        """Save the composite figure and return its path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{suffix}.{fmt}"
        # Use bbox_inches=None to preserve composite layout (avoid tight-cropping empty subplots)
        self.fig.savefig(path, format=fmt, bbox_inches=None)
        plt.close(self.fig)
        return path


# ---------------------------------------------------------------------------
# Pre-built composite figure templates for top-journal main text
# ---------------------------------------------------------------------------

class CompositeFigureTemplates:
    """Ready-to-use composite figure templates."""

    @staticmethod
    def model_performance_overview(output_dir: Path, result: dict,
                                   y_true: list, y_pred: list) -> Path | None:
        """Composite: parity + residual scatter + residual hist + QQ."""
        from .figure_generator import FigureGenerator
        fg = FigureGenerator(output_dir, "tmp")

        cf = CompositeFigure(layout=(2, 2), figsize=(10, 9))
        cf.add_subplot(_plot_parity_on_ax, y_true, y_pred,
                       result.get("metrics", {}).get("r2"))
        cf.add_subplot(_plot_residual_scatter_on_ax, y_true, y_pred)
        cf.add_subplot(_plot_residual_hist_on_ax, y_true, y_pred)
        cf.add_subplot(_plot_qq_on_ax, y_true, y_pred)
        cf.add_overall_title("Model Performance Overview")
        return cf.save(output_dir, "fig02_model_performance")

    @staticmethod
    def interpretability_overview(output_dir: Path, shap_values, features,
                                  feature_names: list[str] | None = None) -> Path | None:
        """Composite: SHAP bar + swarm + dependence + interaction."""
        cf = CompositeFigure(layout=(2, 2), figsize=(12, 10))
        # Each subplot rendered by shap_analyzer functions on provided ax
        # Stub: will be wired once shap_analyzer supports ax-mode
        cf.add_overall_title("Model Interpretability")
        return cf.save(output_dir, "fig04_interpretability")

    @staticmethod
    def data_overview(output_dir: Path, y_values: list,
                      cleaned_y: list | None = None) -> Path | None:
        """Composite: raw distribution + cleaned distribution + other QC."""
        cf = CompositeFigure(layout=(1, 3), figsize=(14, 4.5))
        cf.add_subplot(_plot_distribution_on_ax, y_values, title="Raw Data")
        if cleaned_y is not None:
            cf.add_subplot(_plot_distribution_on_ax, cleaned_y, title="After Cleaning")
        cf.add_overall_title("Data Quality Overview")
        return cf.save(output_dir, "fig01_data_overview")


# ---------------------------------------------------------------------------
# Internal helpers that draw onto a given Axes (for CompositeFigure use)
# ---------------------------------------------------------------------------

def _plot_parity_on_ax(ax, y_true, y_pred, r2=None):
    import numpy as np
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t, y_p = y_t[valid], y_p[valid]
    if len(y_t) == 0:
        ax.set_visible(False)
        return
    lim_min = min(y_t.min(), y_p.min())
    lim_max = max(y_t.max(), y_p.max())
    pad = (lim_max - lim_min) * 0.05
    ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], "k--", lw=1.0)
    ax.scatter(y_t, y_p, c=_PALETTE[0], s=20, alpha=0.6, edgecolors="none")
    ax.set_xlim(lim_min - pad, lim_max + pad)
    ax.set_ylim(lim_min - pad, lim_max + pad)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    title = "Parity Plot"
    if r2 is not None:
        title += f" ($R^2$={r2:.3f})"
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")


def _plot_residual_scatter_on_ax(ax, y_true, y_pred):
    import numpy as np
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t, y_p = y_t[valid], y_p[valid]
    if len(y_t) == 0:
        ax.set_visible(False)
        return
    residuals = y_t - y_p
    ax.scatter(y_p, residuals, c=_PALETTE[0], s=20, alpha=0.6, edgecolors="none")
    ax.axhline(0, color="black", linestyle="--", lw=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Predicted")


def _plot_residual_hist_on_ax(ax, y_true, y_pred):
    import numpy as np
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    residuals = (y_t - y_p)[valid]
    if len(residuals) == 0:
        ax.set_visible(False)
        return
    ax.hist(residuals, bins=max(15, int(len(residuals) ** 0.5)),
            color=_PALETTE[0], edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", lw=1.0)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")


def _plot_qq_on_ax(ax, y_true, y_pred):
    import numpy as np
    from scipy import stats
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    residuals = (y_t - y_p)[valid]
    if len(residuals) < 5:
        ax.set_visible(False)
        return
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(_PALETTE[0])
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color("black")
    ax.set_title("Normal Q–Q Plot")


def _plot_distribution_on_ax(ax, values, title="Distribution"):
    import numpy as np
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        ax.set_visible(False)
        return
    ax.hist(v, bins=20, color=_PALETTE[0], edgecolor="white", alpha=0.8)
    ax.axvline(np.median(v), color="red", linestyle="--", lw=1.2)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)


def _plot_feature_importance_on_ax(ax, importances, feature_names=None, top_k=15):
    import numpy as np
    if importances is None or len(importances) == 0:
        ax.set_visible(False)
        return
    imp = np.array(importances, dtype=float)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(imp))]
    else:
        feature_names = list(feature_names)
    if len(feature_names) < len(imp):
        feature_names += [f"Feature {i}" for i in range(len(feature_names), len(imp))]
    feature_names = feature_names[:len(imp)]
    order = np.argsort(np.abs(imp))[::-1][:top_k][::-1]
    imp_s = imp[order]
    names_s = [feature_names[i] for i in order]
    colors = [_PALETTE[1] if v < 0 else _PALETTE[0] for v in imp_s]
    ax.barh(range(len(imp_s)), imp_s, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(imp_s)))
    ax.set_yticklabels(names_s, fontsize=7)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(imp_s)} Feature Importances")
    ax.axvline(0, color="black", lw=0.5)


def _plot_cv_folds_on_ax(ax, fold_scores, metric_name=r"$R^2$"):
    import numpy as np
    if not fold_scores or len(fold_scores) < 2:
        ax.set_visible(False)
        return
    scores = np.array(fold_scores, dtype=float)
    mean_s, std_s = scores.mean(), scores.std()
    x_pos = np.arange(len(scores))
    colors = [_PALETTE[0] if s >= mean_s else _PALETTE[3] for s in scores]
    ax.bar(x_pos, scores, color=colors, edgecolor="white", width=0.6)
    ax.axhline(mean_s, color="black", linestyle="-", lw=1.2, label=f"Mean = {mean_s:.3f}")
    ax.axhline(mean_s + std_s, color="gray", linestyle="--", lw=0.8)
    ax.axhline(mean_s - std_s, color="gray", linestyle="--", lw=0.8)
    ax.fill_between([-0.5, len(scores) - 0.5], [mean_s - std_s] * 2,
                    [mean_s + std_s] * 2, color="gray", alpha=0.1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Fold {i + 1}" for i in x_pos], fontsize=8)
    ax.set_ylabel(metric_name)
    ax.set_title(f"CV Fold Performance ({mean_s:.3f}±{std_s:.3f})")
    ax.legend(loc="best", frameon=False, fontsize=8)


def _plot_radar_on_ax(ax, multi_model_results, title="Model Comparison"):
    """Radar chart comparing models on normalized metrics.

    Falls back to a grouped bar chart if ax is not polar.
    """
    import numpy as np
    if not multi_model_results or len(multi_model_results) < 2:
        ax.set_visible(False)
        return

    # Check if ax is polar; if not, draw grouped bar chart fallback
    if not hasattr(ax, 'set_theta_offset'):
        _plot_metric_comparison_bars(ax, multi_model_results, title)
        return

    # Use R², Pearson r, -RMSE (normalized), -MAE (normalized)
    metrics = [("R²", "r2", 1), ("Pearson r", "pearson_r", 1),
               ("-RMSE", "rmse", -1), ("-MAE", "mae", -1)]
    labels = [m[0] for m in metrics]
    n_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)

    for idx, model in enumerate(multi_model_results[:6]):  # max 6 models
        vals = []
        for _, key, sign in metrics:
            v = model.get(key, 0)
            if sign == -1:
                v = -v
            vals.append(v)
        vals_np = np.array(vals)
        min_v, max_v = vals_np.min(), vals_np.max()
        if max_v > min_v:
            vals_np = (vals_np - min_v) / (max_v - min_v)
        vals_np = vals_np.tolist()
        vals_np += vals_np[:1]
        ax.plot(angles, vals_np, color=_PALETTE[idx % len(_PALETTE)], linewidth=1.2,
                label=model.get("name", f"M{idx+1}"))
        ax.fill(angles, vals_np, color=_PALETTE[idx % len(_PALETTE)], alpha=0.1)

    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9, y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7, frameon=False)


def _plot_metric_comparison_bars(ax, multi_model_results, title="Model Comparison"):
    """Grouped bar chart fallback for radar chart on non-polar axes."""
    import numpy as np
    metrics = [("R²", "r2"), ("Pearson r", "pearson_r"), ("RMSE", "rmse"), ("MAE", "mae")]
    labels = [m[0] for m in metrics]
    n_models = len(multi_model_results)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for idx, model in enumerate(multi_model_results):
        vals = [model.get(key, 0) for _, key in metrics]
        # Normalize each metric to [0,1] for visualization
        all_vals = [m.get(key, 0) for _, key in metrics for m in multi_model_results]
        metric_min = [min(m.get(key, 0) for m in multi_model_results) for _, key in metrics]
        metric_max = [max(m.get(key, 0) for m in multi_model_results) for _, key in metrics]
        norm_vals = []
        for v, mn, mx in zip(vals, metric_min, metric_max):
            if mx > mn:
                norm_vals.append((v - mn) / (mx - mn))
            else:
                norm_vals.append(0.5)
        ax.bar(x + idx * width - 0.4 + width/2, norm_vals, width,
               color=_PALETTE[idx % len(_PALETTE)], alpha=0.8,
               label=model.get("name", f"M{idx+1}"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Normalized Score")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="best", frameon=False, fontsize=7)
    ax.set_ylim(0, 1.2)


def _plot_pareto_on_ax(ax, multi_model_results, title="Pareto Front"):
    """Pareto front: RMSE vs R². Lower RMSE and higher R² is better."""
    import numpy as np
    if not multi_model_results or len(multi_model_results) < 2:
        ax.set_visible(False)
        return
    names = [m.get("name", f"M{i+1}") for i, m in enumerate(multi_model_results)]
    r2s = np.array([m.get("r2", 0) for m in multi_model_results])
    rmses = np.array([m.get("rmse", 999) for m in multi_model_results])

    # Identify Pareto optimal points
    is_pareto = np.ones(len(r2s), dtype=bool)
    for i in range(len(r2s)):
        for j in range(len(r2s)):
            if i != j and r2s[j] >= r2s[i] and rmses[j] <= rmses[i] and (r2s[j] > r2s[i] or rmses[j] < rmses[i]):
                is_pareto[i] = False
                break

    ax.scatter(rmses[~is_pareto], r2s[~is_pareto], c="lightgray", s=60, alpha=0.6,
               edgecolors="none", label="Dominated")
    ax.scatter(rmses[is_pareto], r2s[is_pareto], c=_PALETTE[3], s=100, alpha=0.9,
               edgecolors="black", linewidths=0.5, marker="*", label="Pareto optimal", zorder=5)

    for i, name in enumerate(names):
        ax.annotate(name, (rmses[i], r2s[i]), textcoords="offset points",
                    xytext=(4, 4), fontsize=6, alpha=0.8)

    ax.set_xlabel("RMSE [%] (lower is better)", fontsize=8)
    ax.set_ylabel(r"$R^2$ (higher is better)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(loc="best", frameon=False, fontsize=8)
