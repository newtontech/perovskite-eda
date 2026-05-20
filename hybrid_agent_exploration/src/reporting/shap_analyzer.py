"""shap_analyzer.py

Comprehensive SHAP analysis suite: bar, swarm, dependence, and interaction plots.
All functions support both standalone figure mode and Axes mode for composite figures.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


class SHAPAnalyzer:
    """Generate SHAP-based interpretability figures."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._has_shap = False
        try:
            import shap
            self._has_shap = True
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # 1. SHAP Bar Plot (mean absolute value)
    # ------------------------------------------------------------------
    def bar_plot(self, shap_values: list | np.ndarray, features: np.ndarray,
                 feature_names: list[str] | None = None,
                 top_k: int = 15, ax: plt.Axes | None = None,
                 filename: str = "shap_bar.png") -> Path | None:
        """SHAP mean absolute value bar plot."""
        if not self._has_shap:
            return None
        import shap
        try:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6.0, max(4.5, 0.3 * top_k)))
                standalone = True
            else:
                fig = ax.figure
                standalone = False

            shap.summary_plot(np.array(shap_values), features,
                              feature_names=feature_names,
                              plot_type="bar", max_display=top_k,
                              show=False, color=_PALETTE[0])
            ax.set_title("SHAP Feature Importance (Mean |SHAP value|)")
            if standalone:
                path = self.output_dir / filename
                fig.savefig(path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                return path
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 2. SHAP Swarm / Beeswarm Plot
    # ------------------------------------------------------------------
    def swarm_plot(self, shap_values: list | np.ndarray, features: np.ndarray,
                   feature_names: list[str] | None = None,
                   top_k: int = 15, ax: plt.Axes | None = None,
                   filename: str = "shap_swarm.png") -> Path | None:
        """SHAP beeswarm/swarm plot showing feature value distributions."""
        if not self._has_shap:
            return None
        import shap
        try:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6.0, max(4.5, 0.3 * top_k)))
                standalone = True
            else:
                fig = ax.figure
                standalone = False

            shap.summary_plot(np.array(shap_values), features,
                              feature_names=feature_names,
                              max_display=top_k, show=False)
            ax.set_title("SHAP Feature Importance Distribution")
            if standalone:
                path = self.output_dir / filename
                fig.savefig(path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                return path
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 3. SHAP Dependence Plot (single feature marginal effect)
    # ------------------------------------------------------------------
    def dependence_plot(self, shap_values: list | np.ndarray, features: np.ndarray,
                        feature_idx: int | str, feature_names: list[str] | None = None,
                        interaction_idx: int | str | None = None,
                        ax: plt.Axes | None = None,
                        filename: str | None = None) -> Path | None:
        """SHAP dependence plot for a single feature.

        Shows how the SHAP value of a feature changes with its raw value.
        Optionally colored by an interaction feature.
        """
        if not self._has_shap:
            return None
        import shap
        try:
            if isinstance(feature_idx, str) and feature_names:
                feature_idx = feature_names.index(feature_idx)
            if interaction_idx is not None and isinstance(interaction_idx, str) and feature_names:
                interaction_idx = feature_names.index(interaction_idx)

            if ax is None:
                fig, ax = plt.subplots(figsize=(5.5, 4.2))
                standalone = True
            else:
                fig = ax.figure
                standalone = False

            shap.dependence_plot(feature_idx, np.array(shap_values), features,
                                 feature_names=feature_names,
                                 interaction_index=interaction_idx,
                                 show=False, ax=ax)
            fname = feature_names[feature_idx] if feature_names and isinstance(feature_idx, int) else str(feature_idx)
            ax.set_title(f"SHAP Dependence: {fname}")
            if standalone:
                fname_out = filename or f"shap_dependence_{fname.replace(' ', '_')}.png"
                path = self.output_dir / fname_out
                fig.savefig(path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                return path
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 4. SHAP Interaction Plot (two-feature heatmap)
    # ------------------------------------------------------------------
    def interaction_plot(self, shap_values: list | np.ndarray, features: np.ndarray,
                         feature_i: int | str, feature_j: int | str,
                         feature_names: list[str] | None = None,
                         ax: plt.Axes | None = None,
                         filename: str | None = None) -> Path | None:
        """Visualize interaction effect between two features.

        Creates a 2D histogram / heatmap of SHAP values binned by two features.
        """
        if not self._has_shap:
            return None
        try:
            if isinstance(feature_i, str) and feature_names:
                feature_i = feature_names.index(feature_i)
            if isinstance(feature_j, str) and feature_names:
                feature_j = feature_names.index(feature_j)

            sv = np.array(shap_values)
            X = np.array(features)
            if sv.ndim == 1:
                sv = sv.reshape(-1, 1)

            # Compute mean absolute SHAP value per 2D bin
            n_bins = 15
            xi, yi = X[:, feature_i], X[:, feature_j]
            xi_bins = np.linspace(xi.min(), xi.max(), n_bins + 1)
            yi_bins = np.linspace(yi.min(), yi.max(), n_bins + 1)
            heatmap = np.zeros((n_bins, n_bins))
            counts = np.zeros((n_bins, n_bins))

            for k in range(len(xi)):
                ix = min(np.digitize(xi[k], xi_bins) - 1, n_bins - 1)
                iy = min(np.digitize(yi[k], yi_bins) - 1, n_bins - 1)
                heatmap[iy, ix] += np.mean(np.abs(sv[k]))
                counts[iy, ix] += 1

            with np.errstate(divide="ignore", invalid="ignore"):
                heatmap = np.divide(heatmap, counts)
                heatmap = np.nan_to_num(heatmap)

            if ax is None:
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                standalone = True
            else:
                fig = ax.figure
                standalone = False

            im = ax.imshow(heatmap, origin="lower", aspect="auto", cmap="RdYlGn_r",
                           extent=[xi_bins[0], xi_bins[-1], yi_bins[0], yi_bins[-1]])
            cbar = fig.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label("Mean |SHAP value|")
            name_i = feature_names[feature_i] if feature_names else f"Feature {feature_i}"
            name_j = feature_names[feature_j] if feature_names else f"Feature {feature_j}"
            ax.set_xlabel(name_i)
            ax.set_ylabel(name_j)
            ax.set_title(f"SHAP Interaction: {name_i} × {name_j}")

            if standalone:
                fname_out = filename or f"shap_interaction_{name_i.replace(' ', '_')}_{name_j.replace(' ', '_')}.png"
                path = self.output_dir / fname_out
                fig.savefig(path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                return path
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 5. Chemical Semantic SHAP: highlight substructures for Morgan bits
    # ------------------------------------------------------------------
    def chemical_semantic_plot(
        self,
        shap_values: list | np.ndarray,
        smiles_list: list[str],
        top_k: int = 10,
        radius: int = 2,
        n_bits: int = 2048,
        filename: str = "shap_chemical_semantic.png",
    ) -> Path | None:
        """Generate a figure showing Top-k SHAP features with chemical substructure highlights.

        For Morgan fingerprints, each feature index corresponds to a bit.
        This method uses bit_interpreter to find activating molecules and
        highlight the substructure responsible for the bit.
        """
        if not smiles_list:
            return None
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[2]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from features.bit_interpreter import explain_top_shap_bits

            highlight_dir = self.output_dir / "shap_highlights"
            highlight_dir.mkdir(parents=True, exist_ok=True)

            explanations = explain_top_shap_bits(
                shap_values=np.array(shap_values),
                feature_names=None,
                smiles_list=smiles_list,
                top_k=top_k,
                radius=radius,
                n_bits=n_bits,
                output_dir=highlight_dir,
            )

            if not explanations:
                return None

            # Build a summary figure: table of Top-k features with descriptions
            fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * top_k)))
            ax.axis("off")

            rows = []
            for exp in explanations:
                desc = exp.get("description", "unknown")
                shap_val = exp.get("mean_abs_shap", 0)
                n_act = exp.get("n_activating", 0)
                rows.append([
                    f"Bit {exp['bit_index']}",
                    f"{shap_val:.3f}",
                    desc,
                    f"{n_act} mols",
                ])

            table = ax.table(
                cellText=rows,
                colLabels=["Feature", "Mean |SHAP|", "Chemical Interpretation", "Activating"],
                loc="center",
                cellLoc="left",
                colColours=["#4472C4"] * 4,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)
            for i in range(4):
                table[(0, i)].set_text_props(color="white", fontweight="bold")

            ax.set_title("Top SHAP Features: Chemical Semantic Interpretation", fontsize=12, fontweight="bold", pad=20)

            path = self.output_dir / filename
            fig.savefig(path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            return path

        except Exception as e:
            print(f"chemical_semantic_plot failed: {e}")
            return None

    # ------------------------------------------------------------------
    # 6. Batch SHAP for multiple models (SI use case)
    # ------------------------------------------------------------------
    def batch_shap_for_models(self, models_data: list[dict],
                              feature_names: list[str] | None = None) -> list[Path]:
        """Generate SHAP swarm + bar for each model in a list.

        models_data: list of dicts with keys 'name', 'shap_values', 'features'
        Returns list of generated figure paths.
        """
        paths = []
        for m in models_data:
            name = m.get("name", "model")
            sv = m.get("shap_values")
            feat = m.get("features")
            if sv is None or feat is None:
                continue
            p = self.swarm_plot(sv, feat, feature_names=feature_names,
                                filename=f"shap_swarm_{name}.png")
            if p:
                paths.append(p)
            p = self.bar_plot(sv, feat, feature_names=feature_names,
                              filename=f"shap_bar_{name}.png")
            if p:
                paths.append(p)
        return paths
