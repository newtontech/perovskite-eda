"""validation.py — Post-experiment validation of pipeline outputs."""

from typing import Any
import os


class ExperimentValidator:
    """Validate that experiment outputs meet minimum quality thresholds."""

    DEFAULT_THRESHOLDS = {
        "min_r2": -1.0,
        "max_rmse": 10.0,
        "min_n_train": 50,
        "min_n_test": 10,
    }

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def validate(self, result: dict[str, Any]) -> dict[str, Any]:
        """Validate a single experiment result dict.

        Returns dict with:
            - passed: bool
            - violations: list[str]
            - warnings: list[str]
        """
        violations = []
        warnings = []

        metrics = result.get("metrics", {})
        n_train = result.get("n_train", 0)
        n_test = result.get("n_test", 0)
        r2 = metrics.get("r2")
        rmse = metrics.get("rmse")

        if r2 is not None and r2 < self.thresholds["min_r2"]:
            violations.append(f"R² = {r2:.4f} < min {self.thresholds['min_r2']}")
        if rmse is not None and rmse > self.thresholds["max_rmse"]:
            violations.append(f"RMSE = {rmse:.4f} > max {self.thresholds['max_rmse']}")
        if n_train < self.thresholds["min_n_train"]:
            violations.append(f"Train size {n_train} < min {self.thresholds['min_n_train']}")
        if n_test < self.thresholds["min_n_test"]:
            violations.append(f"Test size {n_test} < min {self.thresholds['min_n_test']}")

        # Warnings
        if r2 is not None and r2 < 0:
            warnings.append(f"R² = {r2:.4f} is negative (worse than mean baseline)")
        if n_train < 200:
            warnings.append(f"Small training set: n_train = {n_train}")

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
        }

    @staticmethod
    def check_artifact_paths(base_dir: str) -> dict[str, Any]:
        """Verify expected output files exist."""
        expected = ["results.json", "leaderboard.csv", "figures/"]
        found = {name: os.path.exists(os.path.join(base_dir, name)) for name in expected}
        return {
            "all_present": all(found.values()),
            "files": found,
        }
