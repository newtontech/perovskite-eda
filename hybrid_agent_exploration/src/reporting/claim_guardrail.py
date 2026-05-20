"""claim_guardrail.py — Detect and correct over-claims in auto-generated scientific text.

Rules are based on common peer-review feedback patterns for ML-in-chemistry papers.
"""

import re
from typing import Any


# (condition_metric, condition_threshold, overclaim_pattern, replacement)
CLAIM_RULES = [
    # R²-based claims
    ("r2", 0.3, "strong correlation", "modest predictive signal"),
    ("r2", 0.3, "strongly correlated", "weakly to modestly correlated"),
    ("r2", 0.5, "effectively encode", "capture partial"),
    ("r2", 0.5, "highly accurate", "moderately accurate"),
    ("r2", 0.5, "excellent performance", "reasonable performance"),
    ("r2", 0.7, "state-of-the-art", "competitive with literature benchmarks"),
    ("r2", 0.7, "outperforms", "achieves comparable performance to"),
    # Pearson r-based claims
    ("pearson_r", 0.5, "strong linear agreement", "moderate linear agreement"),
    ("pearson_r", 0.5, "strong agreement", "moderate agreement"),
    ("pearson_r", 0.7, "excellent agreement", "good agreement"),
    # RMSE-based claims (lower is better, so inverted logic in code)
    ("rmse", 3.0, "low prediction error", "acceptable prediction error"),
    # P-value based claims
    ("pvalue", 0.05, "significantly outperforms", "shows comparable performance to"),
    ("pvalue", 0.05, "significantly better", "comparable to"),
]


class ClaimGuardrail:
    """Scan and sanitize over-claims in scientific narrative."""

    @staticmethod
    def sanitize_narrative(text: str, metrics: dict[str, Any]) -> str:
        """Replace over-claims with conservative alternatives based on metrics.

        Args:
            text: Narrative text (e.g., Abstract or Results section).
            metrics: Dict with keys like 'r2', 'pearson_r', 'rmse', 'pvalue'.

        Returns:
            Corrected text.
        """
        corrected = text
        warnings = []

        for metric_key, threshold, overclaim, replacement in CLAIM_RULES:
            val = metrics.get(metric_key)
            if val is None:
                continue

            # Determine if condition is met (overclaim is unjustified)
            if metric_key == "rmse":
                # For RMSE, higher values are worse; threshold is "acceptable max"
                condition_met = val > threshold
            elif metric_key == "pvalue":
                # For p-value, > 0.05 means NOT significant
                condition_met = val > threshold
            else:
                # For R², Pearson r: lower than threshold = overclaim
                condition_met = val < threshold

            if condition_met and overclaim.lower() in corrected.lower():
                # Case-insensitive replacement, preserve original case pattern
                pattern = re.compile(re.escape(overclaim), re.IGNORECASE)
                corrected = pattern.sub(replacement, corrected)
                warnings.append(
                    f"ClaimGuardrail: '{overclaim}' → '{replacement}' "
                    f"(because {metric_key}={val:.4f} < threshold={threshold})"
                )

        return corrected, warnings

    @staticmethod
    def check_abstract_against_metrics(abstract: str, metrics: dict[str, Any]) -> dict[str, Any]:
        """Run full check on an abstract and return corrections + summary."""
        corrected, warnings = ClaimGuardrail.sanitize_narrative(abstract, metrics)
        return {
            "original": abstract,
            "corrected": corrected,
            "warnings": warnings,
            "n_issues": len(warnings),
            "passed": len(warnings) == 0,
        }

    @staticmethod
    def get_performance_qualifier(metric_key: str, value: float) -> str:
        """Return an appropriate qualitative descriptor for a metric value."""
        if metric_key == "r2":
            if value < 0:
                return "no predictive power (worse than mean baseline)"
            elif value < 0.3:
                return "modest predictive signal"
            elif value < 0.5:
                return "moderate predictive ability"
            elif value < 0.7:
                return "reasonably strong predictive ability"
            elif value < 0.9:
                return "strong predictive ability"
            else:
                return "excellent predictive ability"
        elif metric_key == "pearson_r":
            abs_r = abs(value)
            if abs_r < 0.3:
                return "weak linear correlation"
            elif abs_r < 0.5:
                return "moderate linear correlation"
            elif abs_r < 0.7:
                return "reasonably strong linear correlation"
            elif abs_r < 0.9:
                return "strong linear correlation"
            else:
                return "very strong linear correlation"
        elif metric_key == "rmse":
            # Context-dependent; for delta_pce in %, <2 is good, >5 is poor
            if value < 1.5:
                return "low prediction error"
            elif value < 3.0:
                return "moderate prediction error"
            elif value < 5.0:
                return "high prediction error"
            else:
                return "very high prediction error"
        return "unknown performance level"
