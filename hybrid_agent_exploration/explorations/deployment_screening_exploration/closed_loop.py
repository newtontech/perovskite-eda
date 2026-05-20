"""Closed-Loop Simulation (D54) — Train → Screen → Validate → Retrain"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)


class ClosedLoopSimulator:
    """Simulate an autonomous discovery loop with experimental feedback."""

    def __init__(
        self,
        screener,
        candidate_pool: pd.DataFrame,
        ground_truth_fn: Optional[Callable[[str], float]] = None,
        noise_std: float = 0.3,
        validation_split: float = 0.2,
        random_state: int = 42,
    ):
        self.screener = screener
        self.candidate_pool = candidate_pool.copy().reset_index(drop=True)
        self.ground_truth_fn = ground_truth_fn
        self.noise_std = noise_std
        self.validation_split = validation_split
        self.random_state = random_state
        self.history: list[dict] = []
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None

    def initialize(self, train_df: pd.DataFrame):
        """Set initial training data and fit screener."""
        self.train_df = train_df.copy()
        # Create a persistent validation set from candidates for stable metrics
        n_val = max(1, int(len(self.candidate_pool) * self.validation_split))
        self.val_df = self.candidate_pool.sample(
            n=n_val, random_state=self.random_state
        ).copy()
        # Remove validation set from candidate pool
        self.candidate_pool = (
            self.candidate_pool.drop(self.val_df.index)
            .reset_index(drop=True)
        )

        self.screener.fit(
            self.train_df, target_col="delta_pce", smiles_col="smiles"
        )
        metrics = self._evaluate()
        self._record(0, metrics, [], [])
        print(
            f"[ClosedLoop] Initialized with {len(self.train_df)} train, "
            f"{len(self.val_df)} val, {len(self.candidate_pool)} candidates"
        )

    def run(
        self, n_iterations: int = 5, k: int = 5, strategy: str = "top_k"
    ) -> list[dict]:
        """Execute the full closed-loop."""
        for it in range(1, n_iterations + 1):
            print(
                f"\n[ClosedLoop] === Iteration {it}/{n_iterations} | "
                f"strategy={strategy} | k={k} ==="
            )
            metrics, selected_smiles, measured_values = self.run_iteration(
                k=k, strategy=strategy
            )
            self._record(it, metrics, selected_smiles, measured_values)
        return self.history

    def run_iteration(
        self, k: int = 5, strategy: str = "top_k"
    ) -> tuple[dict, list[str], list[float]]:
        """Single loop: screen → select top-k → simulate experiment → retrain."""
        if self.candidate_pool.empty:
            print("[ClosedLoop] Candidate pool exhausted.")
            return self._evaluate(), [], []

        # 1. Score remaining candidates
        scores_df = self.screener.score_candidates(
            self.candidate_pool["smiles"].tolist()
        )
        # 2. Select top-k by strategy
        top_k = self.screener.rank_by_strategy(
            scores_df, strategy=strategy, k=k
        )

        selected_smiles = top_k["smiles"].tolist()
        # 3. Simulate experimental validation
        measured_values = self.simulate_experiment(selected_smiles)

        # 4. Add new data to training set
        new_rows = pd.DataFrame(
            {"smiles": selected_smiles, "delta_pce": measured_values}
        )
        self.train_df = pd.concat(
            [self.train_df, new_rows], ignore_index=True
        )

        # Remove selected from candidate pool
        mask = ~self.candidate_pool["smiles"].isin(selected_smiles)
        self.candidate_pool = self.candidate_pool[mask].reset_index(
            drop=True
        )

        # 5. Retrain
        self.screener.fit(
            self.train_df, target_col="delta_pce", smiles_col="smiles"
        )

        metrics = self._evaluate()
        print(
            f"  → Selected {len(selected_smiles)} | Train size: {len(self.train_df)} | "
            f"Val R²: {metrics['val_r2']:.3f} | "
            f"Top-k true mean: {metrics['top_k_true_mean']:.3f}"
        )
        return metrics, selected_smiles, measured_values

    def simulate_experiment(self, smiles_list: list[str]) -> list[float]:
        """Return 'measured' delta_PCE with experimental noise."""
        values = []
        for smi in smiles_list:
            if self.ground_truth_fn is not None:
                true_val = self.ground_truth_fn(smi)
            else:
                # Fallback: look up in original pool if available
                true_val = self._lookup_true(smi)
            noise = np.random.normal(0, self.noise_std)
            values.append(float(true_val + noise))
        return values

    def _lookup_true(self, smiles: str) -> float:
        """Look up ground truth from remaining pool or validation set."""
        for df in (self.candidate_pool, self.val_df):
            if df is not None:
                row = df[df["smiles"] == smiles]
                if not row.empty:
                    return float(row["delta_pce"].iloc[0])
        return 0.0

    def _evaluate(self) -> dict:
        """Evaluate current model on validation set and report top-k stats."""
        if self.val_df is None or self.val_df.empty:
            return {
                "val_r2": 0.0,
                "val_rmse": 0.0,
                "top_k_true_mean": 0.0,
                "top_k_pred_mean": 0.0,
            }

        X_val = self.screener._featurize(self.val_df["smiles"])
        Xs_val = self.screener.scaler.transform(X_val)
        y_val = self.val_df["delta_pce"].values
        y_pred = self.screener.model.predict(Xs_val)

        val_r2 = r2_score(y_val, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        # Evaluate what the top-10 predicted candidates look like in reality
        scores_df = self.screener.score_candidates(
            self.val_df["smiles"].tolist()
        )
        top10 = scores_df.nsmallest(10, "rank")
        # Avoid duplicate SMILES issues
        val_unique = self.val_df.drop_duplicates(subset=["smiles"]).set_index(
            "smiles"
        )
        common = val_unique.index.intersection(top10["smiles"])
        top_k_true_mean = (
            val_unique.loc[common, "delta_pce"].mean()
            if len(common) > 0
            else 0.0
        )

        return {
            "val_r2": float(val_r2),
            "val_rmse": float(val_rmse),
            "top_k_true_mean": float(top_k_true_mean),
            "top_k_pred_mean": float(top10["score"].mean()),
        }

    def _record(
        self,
        iteration: int,
        metrics: dict,
        selected_smiles: list[str],
        measured_values: list[float],
    ):
        self.history.append(
            {
                "iteration": iteration,
                "n_train": len(self.train_df) if self.train_df is not None else 0,
                **metrics,
                "selected_smiles": selected_smiles,
                "measured_values": [float(v) for v in measured_values],
            }
        )

    def plot_history(self, out_path: Path):
        """Save closed-loop trajectory plot."""
        if not self.history:
            return
        hist = pd.DataFrame(self.history)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(
            hist["iteration"], hist["val_r2"], marker="o", color="forestgreen"
        )
        axes[0, 0].set_title("Validation R² over Iterations")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("R²")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(
            hist["iteration"], hist["val_rmse"], marker="o", color="crimson"
        )
        axes[0, 1].set_title("Validation RMSE over Iterations")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(
            hist["iteration"],
            hist["top_k_true_mean"],
            marker="o",
            color="steelblue",
        )
        axes[1, 0].set_title(
            "True Mean ΔPCE of Top-10 Predicted Candidates"
        )
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("ΔPCE")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(
            hist["iteration"], hist["n_train"], marker="o", color="darkorange"
        )
        axes[1, 1].set_title("Training Set Size")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("# Samples")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Closed-Loop Discovery Trajectory (D54)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[Plot] Saved closed-loop trajectory → {out_path}")
