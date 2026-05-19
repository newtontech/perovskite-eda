"""figure_generator.py

Unified figure generation interface for per-experiment reports.
Follows Advanced Materials and materials informatics visualization standards.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Plotting defaults — Advanced Materials style
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Color palette: conservative, colorblind-friendly
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
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "figure.figsize": (5.5, 4.2),
})


class FigureGenerator:
    """Generate publication-quality figures for a single experiment."""

    def __init__(self, output_dir: Path, agent_id: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_id = agent_id
        self._fig_counter = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _next_path(self, suffix: str, fmt: str = "png") -> Path:
        self._fig_counter += 1
        return self.output_dir / f"{self.agent_id}_fig{self._fig_counter:02d}_{suffix}.{fmt}"

    def _save(self, fig: plt.Figure, suffix: str, fmt: str = "png") -> Path:
        path = self._next_path(suffix, fmt)
        fig.savefig(path, format=fmt)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 1. Parity Plot — Predicted vs Actual
    # ------------------------------------------------------------------
    def parity_plot(self, y_true: list, y_pred: list, metric_r2: float | None = None,
                    xlabel: str = "Observed ΔPCE [%]", ylabel: str = "Predicted ΔPCE [%]") -> Path | None:
        """Parity plot with 1:1 reference line and ±10% error bands."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return None
        y_t = np.array(y_true, dtype=float)
        y_p = np.array(y_pred, dtype=float)
        valid = ~(np.isnan(y_t) | np.isnan(y_p))
        if valid.sum() < 5:
            return None
        y_t, y_p = y_t[valid], y_p[valid]

        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        lim_min = min(y_t.min(), y_p.min())
        lim_max = max(y_t.max(), y_p.max())
        pad = (lim_max - lim_min) * 0.05
        lim_min -= pad
        lim_max += pad

        # 1:1 line
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.0, label="1:1 line", zorder=1)
        # ±10% band
        band = max(abs(lim_min), abs(lim_max)) * 0.1
        ax.fill_between([lim_min, lim_max], [lim_min - band, lim_max - band],
                        [lim_min + band, lim_max + band], color="gray", alpha=0.15, label="±10%")

        # Scatter with density coloring
        ax.scatter(y_t, y_p, c=_PALETTE[0], s=25, alpha=0.6, edgecolors="none", zorder=2)

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        title = "Parity Plot"
        if metric_r2 is not None:
            title += f"  ($R^2$={metric_r2:.3f})"
        ax.set_title(title)
        ax.legend(loc="upper left", frameon=False)
        ax.set_aspect("equal", adjustable="box")
        return self._save(fig, "parity")

    # ------------------------------------------------------------------
    # 2. Residual Analysis
    # ------------------------------------------------------------------
    def residual_analysis(self, y_true: list, y_pred: list,
                          xlabel: str = "Predicted ΔPCE [%]") -> list[Path]:
        """Generate residual distribution, residual vs fitted, and QQ plot."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return []
        y_t = np.array(y_true, dtype=float)
        y_p = np.array(y_pred, dtype=float)
        valid = ~(np.isnan(y_t) | np.isnan(y_p))
        y_t, y_p = y_t[valid], y_p[valid]
        if len(y_t) < 5:
            return []

        residuals = y_t - y_p
        paths = []

        # 2a. Residual vs Predicted
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.scatter(y_p, residuals, c=_PALETTE[0], s=25, alpha=0.6, edgecolors="none")
        ax.axhline(0, color="black", linestyle="--", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Residuals [Observed − Predicted]")
        ax.set_title("Residuals vs. Predicted")
        paths.append(self._save(fig, "residual_vs_fitted"))

        # 2b. Residual histogram
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.hist(residuals, bins=max(15, int(len(residuals) ** 0.5)), color=_PALETTE[0],
                edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", lw=1.0)
        ax.set_xlabel("Residuals [Observed − Predicted]")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Distribution")
        paths.append(self._save(fig, "residual_hist"))

        # 2c. QQ plot
        try:
            from scipy import stats
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.get_lines()[0].set_markerfacecolor(_PALETTE[0])
            ax.get_lines()[0].set_markersize(4)
            ax.get_lines()[0].set_alpha(0.6)
            ax.get_lines()[1].set_color("black")
            ax.set_title("Normal Q–Q Plot of Residuals")
            paths.append(self._save(fig, "residual_qq"))
        except Exception:
            pass

        return paths

    # ------------------------------------------------------------------
    # 3. Feature Importance
    # ------------------------------------------------------------------
    def feature_importance(self, importances: list | np.ndarray,
                           feature_names: list[str] | None = None,
                           top_k: int = 20) -> Path | None:
        """Horizontal bar chart of feature importances (model-native or permutation)."""
        if importances is None or len(importances) == 0:
            return None
        imp = np.array(importances, dtype=float)
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(imp))]
        else:
            feature_names = list(feature_names)

        # Pad or trim
        if len(feature_names) < len(imp):
            feature_names += [f"Feature {i}" for i in range(len(feature_names), len(imp))]
        feature_names = feature_names[:len(imp)]

        # Sort by absolute importance
        order = np.argsort(np.abs(imp))[::-1][:top_k][::-1]
        imp_s = imp[order]
        names_s = [feature_names[i] for i in order]

        fig, ax = plt.subplots(figsize=(6.0, max(4.0, 0.25 * len(order))))
        colors = [_PALETTE[1] if v < 0 else _PALETTE[0] for v in imp_s]
        ax.barh(range(len(imp_s)), imp_s, color=colors, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(imp_s)))
        ax.set_yticklabels(names_s, fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {len(imp_s)} Feature Importances")
        ax.axvline(0, color="black", lw=0.5)
        fig.tight_layout()
        return self._save(fig, "feature_importance")

    # ------------------------------------------------------------------
    # 4. CV Fold Performance
    # ------------------------------------------------------------------
    def cv_fold_performance(self, fold_scores: list[float],
                            metric_name: str = r"$R^2$") -> Path | None:
        """Bar chart showing per-fold CV scores with mean ± std overlay."""
        if not fold_scores or len(fold_scores) < 2:
            return None
        scores = np.array(fold_scores, dtype=float)
        mean_s, std_s = scores.mean(), scores.std()

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        x_pos = np.arange(len(scores))
        colors = [_PALETTE[0] if s >= mean_s else _PALETTE[3] for s in scores]
        ax.bar(x_pos, scores, color=colors, edgecolor="white", width=0.6)
        ax.axhline(mean_s, color="black", linestyle="-", lw=1.2, label=f"Mean = {mean_s:.3f}")
        ax.axhline(mean_s + std_s, color="gray", linestyle="--", lw=0.8)
        ax.axhline(mean_s - std_s, color="gray", linestyle="--", lw=0.8)
        ax.fill_between([-0.5, len(scores) - 0.5], [mean_s - std_s] * 2,
                        [mean_s + std_s] * 2, color="gray", alpha=0.1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Fold {i + 1}" for i in x_pos])
        ax.set_ylabel(metric_name)
        ax.set_title(f"Cross-Validation Fold Performance  (mean±std = {mean_s:.3f}±{std_s:.3f})")
        ax.legend(loc="best", frameon=False)
        return self._save(fig, "cv_folds")

    # ------------------------------------------------------------------
    # 5. Feature Correlation Heatmap
    # ------------------------------------------------------------------
    def correlation_heatmap(self, X: np.ndarray,
                            feature_names: list[str] | None = None,
                            max_features: int = 30) -> Path | None:
        """Correlation heatmap of feature matrix (sampled if too large)."""
        if X is None or X.shape[1] < 2:
            return None
        import pandas as pd
        import seaborn as sns

        df = pd.DataFrame(X)
        if feature_names and len(feature_names) == X.shape[1]:
            df.columns = feature_names
        else:
            df.columns = [f"F{i}" for i in range(X.shape[1])]

        # If too many features, select top variance ones
        if df.shape[1] > max_features:
            variances = df.var().sort_values(ascending=False).head(max_features).index
            df = df[variances]

        corr = df.corr()
        fig, ax = plt.subplots(figsize=(max(6, 0.3 * corr.shape[1]), max(5, 0.3 * corr.shape[0])))
        sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.3, cbar_kws={"shrink": 0.7}, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        fig.tight_layout()
        return self._save(fig, "correlation_heatmap")

    # ------------------------------------------------------------------
    # 6. t-SNE / UMAP Feature Space
    # ------------------------------------------------------------------
    def feature_space_projection(self, X: np.ndarray, y: np.ndarray | list | None = None,
                                  method: str = "umap") -> Path | None:
        """2D projection of feature space colored by target value."""
        if X is None or len(X) < 10:
            return None
        try:
            if method == "umap":
                import umap
                reducer = umap.UMAP(n_neighbors=min(15, len(X) - 1), min_dist=0.1, random_state=42)
            else:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, perplexity=min(30, len(X) - 1), random_state=42)
            emb = reducer.fit_transform(X)
        except Exception:
            return None

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        if y is not None:
            y_arr = np.array(y, dtype=float)
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_arr, cmap="viridis", s=20, alpha=0.7, edgecolors="none")
            cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
            cbar.set_label("ΔPCE [%]")
        else:
            ax.scatter(emb[:, 0], emb[:, 1], c=_PALETTE[0], s=20, alpha=0.7, edgecolors="none")
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.set_title(f"Feature Space Projection ({method.upper()})")
        return self._save(fig, f"feature_space_{method}")

    # ------------------------------------------------------------------
    # 7. Error Distribution by Binned Feature
    # ------------------------------------------------------------------
    def error_by_feature_bins(self, y_true: list, y_pred: list,
                              feature_values: list | np.ndarray,
                              feature_name: str = "Baseline PCE [%]",
                              n_bins: int = 5) -> Path | None:
        """Boxplot of absolute errors stratified by a feature's quantile bins."""
        if not y_true or not y_pred or feature_values is None:
            return None
        y_t = np.array(y_true, dtype=float)
        y_p = np.array(y_pred, dtype=float)
        fv = np.array(feature_values, dtype=float)
        valid = ~(np.isnan(y_t) | np.isnan(y_p) | np.isnan(fv))
        y_t, y_p, fv = y_t[valid], y_p[valid], fv[valid]
        if len(y_t) < 10:
            return None

        errors = np.abs(y_t - y_p)
        import pandas as pd
        df = pd.DataFrame({"error": errors, "feature": fv})
        df["bin"] = pd.qcut(df["feature"], q=n_bins, duplicates="drop")

        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        bp = ax.boxplot([group["error"].values for name, group in df.groupby("bin", observed=False)],
                         patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], _PALETTE[:len(bp["boxes"])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticklabels([str(b) for b in df["bin"].cat.categories], rotation=30, ha="right")
        ax.set_xlabel(f"{feature_name} (quantile bins)")
        ax.set_ylabel("Absolute Error |Observed − Predicted|")
        ax.set_title(f"Error Distribution across {feature_name} Bins")
        fig.tight_layout()
        return self._save(fig, "error_by_bins")

    # ------------------------------------------------------------------
    # 8. Learning Curve (placeholder — requires multiple training runs)
    # ------------------------------------------------------------------
    def learning_curve(self, train_sizes: list, train_scores: list, test_scores: list) -> Path | None:
        """Learning curve: performance vs. training set size."""
        if not train_sizes or not train_scores or not test_scores:
            return None
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.plot(train_sizes, train_scores, "o-", color=_PALETTE[0], label="Training", markersize=4)
        ax.plot(train_sizes, test_scores, "s--", color=_PALETTE[1], label="Validation", markersize=4)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(r"$R^2$")
        ax.set_title("Learning Curve")
        ax.legend(frameon=False)
        ax.set_xscale("log")
        return self._save(fig, "learning_curve")

    # ------------------------------------------------------------------
    # 9. Radar / Spider Plot for multi-metric comparison
    # ------------------------------------------------------------------
    def radar_chart(self, metrics: dict[str, float], title: str = "Pipeline Profile") -> Path | None:
        """Radar chart comparing multiple normalized metrics."""
        if not metrics:
            return None
        labels = list(metrics.keys())
        values = list(metrics.values())
        n = len(labels)
        if n < 3:
            return None

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color=_PALETTE[0], alpha=0.25)
        ax.plot(angles, values, color=_PALETTE[0], linewidth=1.5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(title, y=1.08)
        return self._save(fig, "radar")

    # ------------------------------------------------------------------
    # 10. Multi-Model Parity Grid
    # ------------------------------------------------------------------
    def multi_model_parity_grid(self, results: list[dict], ncols: int = 4) -> Path | None:
        """Grid of parity plots for multiple models (top-journal composite figure).

        results: list of dicts with keys 'name', 'y_true', 'y_pred', 'r2'
        """
        if not results:
            return None
        nrows = int(np.ceil(len(results) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.6 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        elif nrows == 1 or ncols == 1:
            axes = np.array(axes).flatten()
        else:
            axes = axes.flatten()

        for idx, res in enumerate(results):
            ax = axes[idx]
            y_t = np.array(res.get("y_true", []), dtype=float)
            y_p = np.array(res.get("y_pred", []), dtype=float)
            name = res.get("name", f"Model {idx+1}")
            r2 = res.get("r2")
            valid = ~(np.isnan(y_t) | np.isnan(y_p))
            y_t, y_p = y_t[valid], y_p[valid]
            if len(y_t) > 0:
                lim_min = min(y_t.min(), y_p.min())
                lim_max = max(y_t.max(), y_p.max())
                pad = (lim_max - lim_min) * 0.05
                ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], "k--", lw=0.8)
                ax.scatter(y_t, y_p, c=_PALETTE[0], s=8, alpha=0.5, edgecolors="none")
                ax.set_xlim(lim_min - pad, lim_max + pad)
                ax.set_ylim(lim_min - pad, lim_max + pad)
            ax.set_xlabel("Observed", fontsize=7)
            ax.set_ylabel("Predicted", fontsize=7)
            title = name
            if r2 is not None:
                title += f"\n$R^2$={r2:.3f}"
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_aspect("equal", adjustable="box")

        # Hide unused subplots
        for idx in range(len(results), len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout(pad=1.0)
        return self._save(fig, "prediction_grid")

    # ------------------------------------------------------------------
    # 11. Multi-Model Performance Comparison Bars
    # ------------------------------------------------------------------
    def performance_comparison_bars(self, results: list[dict],
                                    metrics: list[str] = None) -> Path | None:
        """Side-by-side bar charts for multiple metrics across models (top-journal composite figure).

        results: list of dicts with keys 'name', 'rmse', 'r2', 'pearson_r', 'mae'
        """
        if not results:
            return None
        if metrics is None:
            metrics = ["rmse", "r2", "pearson_r", "mae"]
        metric_labels = {"rmse": "RMSE", "r2": r"$R^2$", "pearson_r": r"$r$", "mae": "MAE"}
        n_metrics = len(metrics)
        fig_width = max(3.5 * n_metrics, 0.18 * len(results) * n_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, 4.8))
        if n_metrics == 1:
            axes = [axes]

        names = [r.get("name", f"M{i+1}") for i, r in enumerate(results)]
        display_names = [f"M{i + 1}" for i in range(len(names))] if len(names) > 12 else names
        x = np.arange(len(names))
        width = 0.6

        for ax, metric in zip(axes, metrics):
            vals = [r.get(metric, 0) for r in results]
            colors = [_PALETTE[0] if v >= np.median(vals) else _PALETTE[3] for v in vals] if metric == "r2" else [_PALETTE[0] if v <= np.median(vals) else _PALETTE[3] for v in vals]
            ax.bar(x, vals, width, color=colors, edgecolor="white")
            ax.set_xticks(x)
            if len(names) > 12:
                ax.set_xticklabels(display_names, rotation=90, ha="center", fontsize=5)
                ax.set_xlabel("Model index")
            else:
                ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(metric_labels.get(metric, metric))
            ax.set_title(metric_labels.get(metric, metric))
            # Annotate best value
            if metric in ("r2", "pearson_r"):
                best_idx = int(np.argmax(vals))
            else:
                best_idx = int(np.argmin(vals))
            ax.annotate(f"{vals[best_idx]:.3f}", xy=(best_idx, vals[best_idx]),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, fontweight="bold")

        fig.tight_layout(pad=1.0)
        return self._save(fig, "performance_comparison")

    # ------------------------------------------------------------------
    # 12. Data Distribution Histogram
    # ------------------------------------------------------------------
    def data_distribution(self, values: list | np.ndarray, xlabel: str = "ΔPCE [%]",
                          title: str = "Data Distribution", bins: int = 20) -> Path | None:
        """Histogram of target variable or feature distribution (top-journal SI figure)."""
        if values is None or len(values) == 0:
            return None
        v = np.array(values, dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return None

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.hist(v, bins=bins, color=_PALETTE[0], edgecolor="white", alpha=0.8)
        ax.axvline(np.median(v), color="red", linestyle="--", lw=1.2, label=f"Median = {np.median(v):.2f}")
        ax.axvline(np.mean(v), color="orange", linestyle="--", lw=1.2, label=f"Mean = {np.mean(v):.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend(frameon=False)
        return self._save(fig, "data_distribution")

    # ------------------------------------------------------------------
    # 13. External Validation Plot
    # ------------------------------------------------------------------
    def external_validation(self, y_true: list, y_pred: list,
                            molecule_names: list[str] | None = None,
                            xlabel: str = "Experimental PCE [%]",
                            ylabel: str = "Predicted PCE [%]") -> Path | None:
        """External validation scatter with relative error annotations (top-journal validation figure)."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return None
        y_t = np.array(y_true, dtype=float)
        y_p = np.array(y_pred, dtype=float)
        valid = ~(np.isnan(y_t) | np.isnan(y_p))
        y_t, y_p = y_t[valid], y_p[valid]
        if len(y_t) < 2:
            return None

        rel_errors = np.abs((y_p - y_t) / y_t) * 100

        fig, ax = plt.subplots(figsize=(6.0, 5.5))
        lim_min = min(y_t.min(), y_p.min())
        lim_max = max(y_t.max(), y_p.max())
        pad = (lim_max - lim_min) * 0.1
        ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], "k--", lw=1.0)
        ax.fill_between([lim_min - pad, lim_max + pad],
                        [lim_min - pad] * 2,
                        [lim_max + pad] * 2,
                        color="gray", alpha=0.08)

        scatter = ax.scatter(y_t, y_p, c=rel_errors, cmap="RdYlGn_r", s=80,
                             edgecolors="black", linewidths=0.3, vmin=0, vmax=20)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label("Relative Error [%]")

        # Annotate each point with name and relative error
        names = molecule_names or [f"M{i+1}" for i in range(len(y_t))]
        for i, (xt, xp, re) in enumerate(zip(y_t, y_p, rel_errors)):
            ax.annotate(f"{names[i]}\n({re:.1f}%)", (xt, xp), textcoords="offset points",
                        xytext=(5, 5), fontsize=6, alpha=0.8)

        ax.set_xlim(lim_min - pad, lim_max + pad)
        ax.set_ylim(lim_min - pad, lim_max + pad)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("External Validation")
        ax.set_aspect("equal", adjustable="box")
        return self._save(fig, "external_validation")
