"""per_experiment_report.py

Generate a standalone scientific report for a single pipeline experiment,
structured following Advanced Materials journal standards.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

from .figure_generator import FigureGenerator
from .image_embedder import embed_markdown_images


# ---------------------------------------------------------------------------
# Layer metadata for narrative generation
# ---------------------------------------------------------------------------
LAYER_DESCRIPTIONS = {
    "L1": {
        "agentic_veryloose": {
            "name": "Agentic Very-Loose Cleaning",
            "philosophy": "Maximally inclusive data curation",
            "hypothesis": "Boundary cases and noisy observations contain signal that strict filtering discards; additive/modulator effects are robustly detectable even in lower-quality data.",
            "risk": "May retain systematic measurement errors or duplicate structures.",
        },
        "agentic_standard": {
            "name": "Agentic Standard Cleaning",
            "philosophy": "Balanced quality-control filtering",
            "hypothesis": "Moderate filtering removes obvious outliers while preserving sufficient sample size for reliable ML training.",
            "risk": "May exclude marginally valid but informative observations.",
        },
        "agentic_strict": {
            "name": "Agentic Strict Cleaning",
            "philosophy": "High-fidelity data only",
            "hypothesis": "Only peer-reviewed, fully characterized devices yield trustworthy structure–property relationships.",
            "risk": "Severely reduced sample size may cause overfitting.",
        },
        "traditional": {
            "name": "Traditional Cleaning",
            "philosophy": "Legacy heuristic-based filtering",
            "hypothesis": "Established domain heuristics effectively remove physically implausible entries.",
            "risk": "Heuristics may not capture all data-quality dimensions.",
        },
    },
    "L2": {
        "F21_rdkit_basic": {
            "name": "RDKit Basic Descriptors",
            "description": "15 physicochemical descriptors (MW, LogP, TPSA, H-bond donors/acceptors, rotatable bonds).",
            "hypothesis": "Global molecular properties dominate additive performance; substructure details are secondary.",
        },
        "F22_maccs": {
            "name": "MACCS Keys (166-bit)",
            "description": "Molecular ACCess System keys capturing common substructural patterns.",
            "hypothesis": "Presence/absence of specific pharmacophore-like substructures is the primary driver of PCE modulation.",
        },
        "F22_ecfp4": {
            "name": "ECFP4 (2048-bit)",
            "description": "Extended-Connectivity Fingerprints with radius 2.",
            "hypothesis": "Local topological environments up to 4 bonds capture additive–perovskite interaction motifs.",
        },
        "F22_ecfp6": {
            "name": "ECFP6 (2048-bit)",
            "description": "Extended-Connectivity Fingerprints with radius 3.",
            "hypothesis": "Larger topological neighborhoods encode long-range electronic effects relevant to interface passivation.",
        },
        "F22_atom_pair": {
            "name": "Atom Pair Fingerprints (2048-bit)",
            "description": "Topological atom-pair distances.",
            "hypothesis": "Spatial relationships between heteroatoms govern donor/acceptor behavior at the perovskite surface.",
        },
        "F22_topological_torsion": {
            "name": "Topological Torsion (2048-bit)",
            "description": "Four-atom topological torsion descriptors.",
            "hypothesis": "Torsion angles and conformational flexibility influence packing and interface energetics.",
        },
        "F22_krfp": {
            "name": "KRFP (4860-bit)",
            "description": "Klekota–Roth functional group fingerprints.",
            "hypothesis": "Fine-grained functional group definitions improve discrimination among chemically similar additives.",
        },
    },
    "L3": {
        "M31_random_forest": {
            "name": "Random Forest",
            "description": "Ensemble of decision trees with bagging.",
            "hypothesis": "Additive-structure effects are highly non-linear and involve feature interactions best captured by tree ensembles.",
            "strengths": ["Robust to overfitting on medium-sized datasets", "Handles mixed feature types", "Provides native importances"],
        },
        "M31_xgboost": {
            "name": "XGBoost",
            "description": "Gradient-boosted trees with regularization.",
            "hypothesis": "Sequential error correction yields superior predictive accuracy for structured molecular data.",
            "strengths": ["High accuracy", "Built-in regularization", "Efficient handling of sparsity"],
        },
        "M31_lightgbm": {
            "name": "LightGBM",
            "description": "Histogram-based gradient boosting.",
            "hypothesis": "Leaf-wise tree growth efficiently models complex feature interactions in high-dimensional fingerprint spaces.",
            "strengths": ["Fast training", "Memory efficient", "Good for large feature spaces"],
        },
        "M31_svr": {
            "name": "Support Vector Regression",
            "description": "Kernel-based regression with ε-insensitive loss.",
            "hypothesis": "A smooth kernel function in a transformed feature space adequately captures structure–property trends.",
            "strengths": ["Good generalization in low-data regimes", "Theoretically grounded"],
        },
        "M31_knn": {
            "name": "k-Nearest Neighbors",
            "description": "Instance-based prediction from nearest training analogues.",
            "hypothesis": "Structurally similar additives yield similar PCE modulation (chemical analogy principle).",
            "strengths": ["Simple", "Interpretable via nearest neighbors"],
        },
    },
    "L4": {
        "E42_random_split": {
            "name": "Random Train/Test Split",
            "description": "80/20 random split with fixed seed.",
            "note": "Provides a single held-out test estimate; susceptible to data leakage if duplicates exist.",
        },
        "E43_5fold_cv": {
            "name": "5-Fold Cross-Validation",
            "description": "Stratified-like k-fold CV (shuffled, seed=42).",
            "note": "Reduces variance in performance estimate; no held-out test set for final model.",
        },
        "E45_shap": {
            "name": "SHAP Interpretability Analysis",
            "description": "Random split plus SHAP feature attribution.",
            "note": "Enables post-hoc model interpretation but adds computational cost.",
        },
    },
    "L5": {
        "D53_top_k": {
            "name": "Top-k Virtual Screening",
            "description": "Rank candidates by predicted PCE boost and return top 20.",
        },
        "D54_report_only": {
            "name": "Report Only",
            "description": "No downstream screening performed; model training and evaluation only.",
        },
    },
}

# Literature baselines from AGENTS.md
LITERATURE_BASELINES = {
    "autogluon_weighted_ensemble": {"r2": 0.1624, "source": "AutoGluon WeightedEnsemble_L2, 5-fold CV", "note": "Baseline using basic chemical descriptors."},
    "autogluon_rf": {"r2": 0.1375, "source": "AutoGluon RandomForestMSE_BAG_L1, 5-fold CV", "note": "Baseline RF performance."},
    "multi_agent_best": {"r2": 0.2962, "source": "Multi-Agent Exploration (this work), MACCS + RF, random split", "note": "Best prior cross-layer result."},
}


class HypothesisGenerator:
    """Generate scientific narrative for a given pipeline configuration."""

    @staticmethod
    def generate(config: dict) -> dict[str, str]:
        l1_id = config.get("layer1", {}).get("method_id", "unknown")
        l2_id = config.get("layer2", {}).get("method_id", "unknown")
        l3_id = config.get("layer3", {}).get("method_id", "unknown")
        l4_id = config.get("layer4", {}).get("method_id", "unknown")
        l5_id = config.get("layer5", {}).get("method_id", "unknown")
        target = config.get("target", "delta_pce")
        baseline_feat = config.get("baseline_as_feature", False)

        l1 = LAYER_DESCRIPTIONS["L1"].get(l1_id, {})
        l2 = LAYER_DESCRIPTIONS["L2"].get(l2_id, {})
        l3 = LAYER_DESCRIPTIONS["L3"].get(l3_id, {})
        l4 = LAYER_DESCRIPTIONS["L4"].get(l4_id, {})
        l5 = LAYER_DESCRIPTIONS["L5"].get(l5_id, {})

        parts = []
        parts.append(f"**Data Curation Hypothesis**: {l1.get('hypothesis', 'No hypothesis available.')}")
        parts.append(f"**Representation Hypothesis**: {l2.get('hypothesis', 'No hypothesis available.')}")
        parts.append(f"**Modeling Hypothesis**: {l3.get('hypothesis', 'No hypothesis available.')}")
        if baseline_feat:
            parts.append("**Baseline-as-Feature Hypothesis**: Including the baseline PCE (without modulator) as an input feature captures device-to-device variability and allows the model to learn additive-specific ΔPCE effects rather than absolute PCE.")

        combined = " ".join(parts)

        title = f"{l3.get('name', l3_id)} Prediction of {target.upper()} using {l2.get('name', l2_id)} with {l1.get('name', l1_id)}"

        return {
            "title": title,
            "hypothesis": combined,
            "l1_name": l1.get("name", l1_id),
            "l2_name": l2.get("name", l2_id),
            "l3_name": l3.get("name", l3_id),
            "l4_name": l4.get("name", l4_id),
            "l5_name": l5.get("name", l5_id),
            "l1_risk": l1.get("risk", ""),
            "l3_strengths": l3.get("strengths", []),
        }


class ResultsInterpreter:
    """Interpret numeric results into scientific prose."""

    @staticmethod
    def interpret(metrics: dict, config: dict, n_samples: int, n_features: int) -> dict[str, str]:
        r2 = metrics.get("r2")
        rmse = metrics.get("rmse")
        r2_std = metrics.get("r2_std")
        strategy = metrics.get("strategy", "unknown")

        interpretations = {}

        # Performance prose
        perf_lines = []
        if r2 is not None:
            if r2 > 0.25:
                perf_lines.append(f"The model achieves a strong coefficient of determination ($R^2$ = {r2:.3f}), indicating that the chosen feature representation effectively encodes the molecular determinants of PCE modulation.")
            elif r2 > 0.10:
                perf_lines.append(f"The model attains a moderate $R^2$ = {r2:.3f}, suggesting partial capture of structure–property relationships but leaving substantial variance unexplained.")
            elif r2 > 0:
                perf_lines.append(f"With $R^2$ = {r2:.3f}, the model shows weak but positive predictive signal, likely limited by insufficient feature discriminability or high data noise.")
            else:
                perf_lines.append(f"The model fails to generalize ($R^2$ = {r2:.3f}), indicating that the chosen feature–model pairing does not capture relevant physicochemical trends in this dataset.")

        if rmse is not None:
            perf_lines.append(f"The root-mean-square error (RMSE = {rmse:.3f} %) quantifies the typical prediction deviation in absolute PCE units.")

        if r2_std is not None and strategy in ("E43_5fold_cv", "E43_10fold_cv"):
            perf_lines.append(f"Cross-validation standard deviation ($R^2_{{std}}$ = {r2_std:.3f}) across folds suggests {'low' if r2_std < 0.05 else 'moderate' if r2_std < 0.15 else 'high'} sensitivity to data partitioning.")

        interpretations["performance"] = " ".join(perf_lines)

        # Comparison with baselines
        comp_lines = []
        best_prior = LITERATURE_BASELINES["multi_agent_best"]["r2"]
        if r2 is not None:
            if r2 > best_prior:
                comp_lines.append(f"This result **exceeds** the prior best cross-layer benchmark ($R^2$ = {best_prior:.3f}), representing a significant advancement in predictive performance for this task.")
            elif r2 > LITERATURE_BASELINES["autogluon_weighted_ensemble"]["r2"]:
                comp_lines.append(f"While below the current best cross-layer result ($R^2$ = {best_prior:.3f}), this configuration **outperforms** the AutoGluon ensemble baseline ($R^2$ = {LITERATURE_BASELINES['autogluon_weighted_ensemble']['r2']:.3f}), demonstrating the value of tailored feature–model pairing.")
            else:
                comp_lines.append(f"This result falls below established baselines, suggesting that the specific combination of methods in this pipeline is suboptimal for the present dataset.")
        interpretations["comparison"] = " ".join(comp_lines)

        # Complexity note
        interpretations["complexity"] = f"The pipeline processes **{n_samples:,}** samples with **{n_features}** input features, yielding a sample-to-feature ratio of ~{n_samples // max(n_features, 1):,}:1."

        return interpretations


class InnovationAssessor:
    """Assess the scientific innovation of a pipeline configuration."""

    @staticmethod
    def assess(config: dict, metrics: dict) -> dict[str, str]:
        l1_id = config.get("layer1", {}).get("method_id", "")
        l2_id = config.get("layer2", {}).get("method_id", "")
        l3_id = config.get("layer3", {}).get("method_id", "")
        l4_id = config.get("layer4", {}).get("method_id", "")
        baseline_feat = config.get("baseline_as_feature", False)
        r2 = metrics.get("r2")

        innovations = []
        if baseline_feat:
            innovations.append("Incorporating baseline PCE as a feature is a domain-informed design choice that decouples device variability from additive effects.")
        if "maccs" in l2_id.lower():
            innovations.append("MACCS fingerprints encode pharmacophore-like substructures, a representation rarely benchmarked for PSC additive screening.")
        if l3_id == "M31_svr" and "maccs" in l2_id.lower():
            innovations.append("The combination of sparse binary fingerprints with kernel regression explores a non-tree-based hypothesis space.")
        if l1_id == "agentic_veryloose":
            innovations.append("The very-loose cleaning strategy challenges the assumption that strict data curation is always beneficial for ML.")
        if l4_id == "E45_shap":
            innovations.append("Explicit SHAP-based interpretability moves beyond black-box prediction toward mechanistic understanding.")

        if not innovations:
            innovations.append("This pipeline represents a conventional but rigorously evaluated configuration, serving as an important reference point in the broader exploration landscape.")

        score_label = "strong" if (r2 is not None and r2 > 0.25) else "moderate" if (r2 is not None and r2 > 0.10) else "limited"

        return {
            "innovations": " ".join(innovations),
            "score_label": score_label,
        }


class NextStepRecommender:
    """Recommend next exploration directions based on current results."""

    @staticmethod
    def recommend(config: dict, metrics: dict) -> list[str]:
        l2_id = config.get("layer2", {}).get("method_id", "")
        l3_id = config.get("layer3", {}).get("method_id", "")
        r2 = metrics.get("r2")
        recs = []

        if r2 is not None and r2 > 0.20:
            recs.append(f"The strong performance of **{l2_id}** + **{l3_id}** suggests further exploration within this feature family: test related fingerprints (e.g., AP, TT, KRFP) or hybrid concatenations.")
            recs.append("Consider increasing evaluation rigor: scaffold split or temporal split to assess true generalization beyond random partitioning.")
            recs.append("If interpretability is critical, perform explicit SHAP analysis on the best-performing model to identify molecular motifs driving high ΔPCE.")
            recs.append("Virtual screening (Layer 5) using this trained model on purchasable molecule libraries (PubChem, Enamine) is now scientifically justified.")
        elif r2 is not None and r2 > 0.05:
            recs.append(f"Moderate performance suggests the feature representation (**{l2_id}**) may be partially informative but insufficient; try richer descriptors (RDKit full, Mordred) or learned embeddings (Uni-Mol, JTVAE).")
            recs.append(f"Model **{l3_id}** may be underfitting; hyperparameter optimization (Optuna, GridSearch) could unlock additional performance.")
            recs.append("Examine residual patterns for systematic errors (e.g., underprediction for high-PCE devices), which may indicate missing physical descriptors.")
        else:
            recs.append(f"Poor performance indicates a fundamental mismatch between **{l2_id}** and **{l3_id}** for this task; consider switching to a different feature class or model family.")
            recs.append("Re-evaluate the data cleaning strategy: overly loose filtering may introduce noise, while overly strict filtering may remove informative outliers.")
            recs.append("Benchmark against a simple baseline (e.g., predicting mean ΔPCE) to confirm the model is learning anything beyond trivial trends.")

        recs.append("In all cases, external experimental validation on 3–5 top-predicted candidates remains the gold standard for PSC additive discovery.")
        return recs


class PerExperimentReport:
    """Generate a full scientific report (Markdown) for one experiment."""

    def __init__(self, result: dict, artifacts: dict | None = None,
                 output_dir: Path | str = "results/reports/per_experiment",
                 embed_images: bool = False):
        self.result = result
        self.config = result.get("config", {})
        self.metrics = result.get("metrics", {})
        self.agent_id = result.get("agent_id", "unknown")
        self.n_samples = result.get("n_samples", 0)
        self.n_features = result.get("n_features", 0)
        self.duration = result.get("duration_sec", 0)
        self.artifacts = artifacts or {}
        self.output_dir = Path(output_dir)
        self.embed_images = embed_images
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_gen = FigureGenerator(self.output_dir / "figures", self.agent_id)

    def generate(self) -> Path:
        """Generate the full Markdown report and return its path."""
        # Generate figures first
        figure_paths = self._generate_figures()

        # Generate narrative sections
        hypo = HypothesisGenerator.generate(self.config)
        interp = ResultsInterpreter.interpret(self.metrics, self.config, self.n_samples, self.n_features)
        innov = InnovationAssessor.assess(self.config, self.metrics)
        recs = NextStepRecommender.recommend(self.config, self.metrics)

        # Assemble report
        lines = []
        lines.append(f"# {hypo['title']}")
        lines.append("")
        lines.append(f"**Experiment ID**: `{self.agent_id}`  ")
        lines.append(f"**Date**: Auto-generated  ")
        lines.append(f"**Status**: {self.result.get('status', 'unknown').upper()}")
        lines.append("")

        # Abstract
        lines.append("## Abstract")
        lines.append(self._generate_abstract(hypo, interp))
        lines.append("")

        # Keywords
        lines.append("**Keywords**: perovskite solar cells, machine learning, molecular additives, "
                     f"{hypo['l2_name'].lower()}, {hypo['l3_name'].lower()}, structure–property relationship, "
                     "high-throughput screening")
        lines.append("")

        # Motivation
        lines.append("## 1. Scientific Motivation and Hypothesis")
        lines.append("")
        lines.append("### 1.1 Research Context")
        lines.append(
            "Accurate prediction of power-conversion-efficiency (PCE) modulation by molecular additives "
            "is a central challenge in perovskite solar-cell (PSC) materials design. "
            "Traditional Edisonian experimentation is too slow to explore the vast chemical space of "
            "potential passivators, HTMs, and anti-solvent additives. "
            "Machine learning offers a data-driven route, yet its utility depends critically on the "
            "joint choices of data curation, molecular representation, model architecture, and evaluation protocol. "
            "This experiment tests one specific cross-layer hypothesis within a broader multi-agent exploration."
        )
        lines.append("")
        lines.append("### 1.2 Pipeline Design Rationale")
        lines.append(f"- **Layer 1 (Data)**: {hypo['l1_name']} — {LAYER_DESCRIPTIONS['L1'].get(self.config.get('layer1', {}).get('method_id', ''), {}).get('philosophy', '')}")
        l2_desc = LAYER_DESCRIPTIONS["L2"].get(self.config.get("layer2", {}).get("method_id", ""), {})
        lines.append(f"- **Layer 2 (Features)**: {hypo['l2_name']} — {l2_desc.get('description', '')}")
        l3_desc = LAYER_DESCRIPTIONS["L3"].get(self.config.get("layer3", {}).get("method_id", ""), {})
        lines.append(f"- **Layer 3 (Model)**: {hypo['l3_name']} — {l3_desc.get('description', '')}")
        lines.append(f"- **Layer 4 (Evaluation)**: {hypo['l4_name']}")
        lines.append(f"- **Layer 5 (Screening)**: {hypo['l5_name']}")
        if self.config.get("baseline_as_feature"):
            lines.append("- **Special Design**: Baseline PCE included as an input feature to isolate additive-specific effects.")
        lines.append("")
        lines.append("### 1.3 Hypothesis Statement")
        lines.append(hypo["hypothesis"])
        lines.append("")

        # Methods
        lines.append("## 2. Experimental Methods")
        lines.append("")
        lines.append("### 2.1 Data Strategy")
        lines.append(f"- Cleaning strategy: **{hypo['l1_name']}**")
        lines.append(f"- Final sample size: **{self.n_samples:,}** devices")
        lines.append(f"- Target variable: **{self.config.get('target', 'delta_pce').upper()}** (PCE with modulator minus PCE without modulator)")
        lines.append("")
        lines.append("### 2.2 Feature Engineering")
        lines.append(f"- Method: {hypo['l2_name']}")
        lines.append(f"- Dimensionality: **{self.n_features}** features")
        lines.append("- Preprocessing: StandardScaler (zero mean, unit variance) applied before model training.")
        lines.append("")
        lines.append("### 2.3 Model Architecture")
        lines.append(f"- Algorithm: {hypo['l3_name']}")
        if hypo.get("l3_strengths"):
            lines.append(f"- Strengths: {', '.join(hypo['l3_strengths'])}")
        lines.append("")
        lines.append("### 2.4 Evaluation Protocol")
        lines.append(f"- Strategy: {hypo['l4_name']}")
        lines.append(f"- Execution time: {self.duration:.1f} s")
        lines.append("")
        lines.append("### 2.5 Screening Strategy")
        lines.append(f"- {hypo['l5_name']}")
        lines.append("")

        # Results
        lines.append("## 3. Results")
        lines.append("")
        lines.append("### 3.1 Model Performance Metrics")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        if self.metrics.get("r2") is not None:
            lines.append(f"| $R^2$ | {self.metrics['r2']:.4f} |")
        if self.metrics.get("r2_std") is not None:
            lines.append(f"| $R^2_{{std}}$ | {self.metrics['r2_std']:.4f} |")
        if self.metrics.get("rmse") is not None:
            lines.append(f"| RMSE | {self.metrics['rmse']:.4f} % |")
        lines.append("")
        lines.append(interp["performance"])
        lines.append("")
        lines.append(interp["complexity"])
        lines.append("")

        # Figures
        if figure_paths:
            lines.append("### 3.2 Visual Analysis")
            lines.append("")
            for fp in figure_paths:
                rel = Path(fp).name
                lines.append(f"![{rel}](figures/{rel})")
                lines.append("")

        # Discussion
        lines.append("## 4. Discussion")
        lines.append("")
        lines.append("### 4.1 Interpretation of Key Findings")
        lines.append(interp["performance"])
        lines.append("")
        lines.append(interp["comparison"])
        lines.append("")
        lines.append("### 4.2 Innovation Assessment")
        lines.append(innov["innovations"])
        lines.append(f"Overall, this pipeline demonstrates **{innov['score_label']}** innovation value.")
        lines.append("")
        if hypo.get("l1_risk"):
            lines.append("### 4.3 Limitations and Caveats")
            lines.append(f"- Data curation risk: {hypo['l1_risk']}")
            lines.append(f"- The sample-to-feature ratio (~{self.n_samples // max(self.n_features, 1):,}:1) {'is adequate' if self.n_samples > 10 * self.n_features else 'is low'} for reliable generalization.")
            lines.append("- Random-split evaluation may overestimate performance if structural duplicates span train/test boundaries.")
            lines.append("")

        # Conclusions
        lines.append("## 5. Conclusions and Future Directions")
        lines.append("")
        lines.append("### 5.1 Summary")
        summary = f"This experiment evaluated a **{hypo['l3_name']}** model trained on **{hypo['l2_name']}** features derived from **{hypo['l1_name']}** data. "
        if self.metrics.get("r2") is not None:
            summary += f"The pipeline achieved $R^2$ = {self.metrics['r2']:.3f}, {interp['comparison'][:200].lower()}..."
        lines.append(summary)
        lines.append("")
        lines.append("### 5.2 Recommendations for Next-Step Exploration")
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        # Supporting Information
        lines.append("## Supporting Information")
        lines.append("")
        lines.append("### S1. Detailed Configuration")
        lines.append("```json")
        lines.append(json.dumps(self.config, indent=2, default=str))
        lines.append("```")
        lines.append("")
        lines.append("### S2. Complete Metrics")
        lines.append("```json")
        lines.append(json.dumps(self.metrics, indent=2, default=str))
        lines.append("```")
        lines.append("")
        if self.artifacts:
            lines.append("### S3. Artifact Summary")
            for key in self.artifacts:
                val = self.artifacts[key]
                if isinstance(val, list):
                    lines.append(f"- **{key}**: array of length {len(val)}")
                elif isinstance(val, np.ndarray):
                    lines.append(f"- **{key}**: ndarray of shape {val.shape}")
                else:
                    lines.append(f"- **{key}**: {type(val).__name__}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Report auto-generated by Hybrid Agent Exploration system. Experiment ID: {self.agent_id}*")

        report_text = "\n".join(lines)
        report_path = self.output_dir / f"{self.agent_id}_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        if self.embed_images:
            embed_markdown_images(report_path)
        return report_path

    def _generate_abstract(self, hypo: dict, interp: dict) -> str:
        r2 = self.metrics.get("r2")
        rmse = self.metrics.get("rmse")
        parts = [
            f"We report a machine-learning pipeline for predicting perovskite solar-cell PCE modulation by molecular additives.",
            f"The pipeline combines {hypo['l1_name'].lower()}, {hypo['l2_name'].lower()}, and {hypo['l3_name'].lower()}.",
        ]
        if self.config.get("baseline_as_feature"):
            parts.append("Baseline device PCE is incorporated as an input feature to isolate additive-specific effects.")
        if r2 is not None:
            parts.append(f"The model achieves $R^2$ = {r2:.3f}")
            if rmse is not None:
                parts.append(f"and RMSE = {rmse:.3f} %")
        parts.append(interp["comparison"])
        parts.append("These findings contribute to the systematic exploration of cross-layer design choices for PSC additive screening.")
        return " ".join(parts)

    def _generate_figures(self) -> list[Path]:
        """Generate all figures and return their paths."""
        paths = []
        y_true = self.artifacts.get("y_true")
        y_pred = self.artifacts.get("y_pred")
        r2 = self.metrics.get("r2")

        if y_true and y_pred:
            p = self.fig_gen.parity_plot(y_true, y_pred, metric_r2=r2)
            if p:
                paths.append(p)
            paths.extend(self.fig_gen.residual_analysis(y_true, y_pred))

        fi = self.artifacts.get("feature_importances")
        if fi:
            p = self.fig_gen.feature_importance(fi)
            if p:
                paths.append(p)

        cv_scores = self.artifacts.get("cv_scores_per_fold")
        if cv_scores:
            p = self.fig_gen.cv_fold_performance(cv_scores)
            if p:
                paths.append(p)

        return paths
