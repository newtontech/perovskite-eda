"""
metrics.py
==========
Evaluation metrics with uncertainty quantification for regression tasks.

Implements E45:
- R², RMSE, MAE
- Uncertainty estimates (ensemble variance, prediction intervals)
- Confidence-based calibration checks
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    return float(r2_score(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (safe against division by zero)."""
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Uncertainty quantification
# ---------------------------------------------------------------------------

class EnsembleUncertainty:
    """
    Uncertainty from an ensemble of predictions (e.g. k-fold models or
    RandomForest trees).  Returns mean prediction + variance-based uncertainty.
    """

    @staticmethod
    def from_predictions(
        predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        predictions : np.ndarray, shape (n_estimators, n_samples)
            Individual predictions from ensemble members.

        Returns
        -------
        mean_pred : np.ndarray, shape (n_samples,)
        std_pred  : np.ndarray, shape (n_samples,)
        pred_interval : np.ndarray, shape (n_samples, 2)  [lower, upper] at 95%
        """
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0, ddof=1)
        # Normal approximation 95% interval
        lower = mean_pred - 1.96 * std_pred
        upper = mean_pred + 1.96 * std_pred
        pred_interval = np.stack([lower, upper], axis=1)
        return mean_pred, std_pred, pred_interval

    @staticmethod
    def negative_log_likelihood(
        y_true: np.ndarray,
        mean_pred: np.ndarray,
        std_pred: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """
        Gaussian negative log-likelihood (lower is better).
        Measures how well the uncertainty calibration fits the data.
        """
        std_pred = np.clip(std_pred, eps, None)
        nll = 0.5 * np.mean(
            np.log(2 * np.pi * std_pred**2)
            + ((y_true - mean_pred) ** 2) / (std_pred**2)
        )
        return float(nll)

    @staticmethod
    def coverage(
        y_true: np.ndarray,
        pred_interval: np.ndarray,
    ) -> float:
        """
        Fraction of true values that fall inside the prediction interval.
        For a well-calibrated 95% interval this should be ~0.95.
        """
        inside = (y_true >= pred_interval[:, 0]) & (y_true <= pred_interval[:, 1])
        return float(np.mean(inside))

    @staticmethod
    def interval_width(
        pred_interval: np.ndarray,
    ) -> float:
        """Mean width of prediction intervals."""
        return float(np.mean(pred_interval[:, 1] - pred_interval[:, 0]))


class BootstrapUncertainty:
    """
    Bootstrap-based uncertainty: resample predictions with replacement
    to estimate sampling variance.
    """

    def __init__(self, n_bootstrap: int = 1000, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def estimate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn,
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Bootstrap a metric to obtain mean, std, and 95% CI.

        Returns
        -------
        mean, std, (ci_lower, ci_upper)
        """
        rng = np.random.RandomState(self.random_state)
        n = len(y_true)
        scores = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        scores = np.array(scores)
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1))
        ci_lower = float(np.percentile(scores, 2.5))
        ci_upper = float(np.percentile(scores, 97.5))
        return mean, std, (ci_lower, ci_upper)


# ---------------------------------------------------------------------------
# Aggregator for CV fold results
# ---------------------------------------------------------------------------

def aggregate_cv_results(
    fold_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate a list of per-fold metric dicts into mean ± std.

    Returns
    -------
    dict: {metric_name: {'mean': ..., 'std': ..., 'min': ..., 'max': ...}}
    """
    if not fold_results:
        return {}
    metrics = list(fold_results[0].keys())
    aggregated = {}
    for metric in metrics:
        values = np.array([fold[metric] for fold in fold_results])
        aggregated[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }
    return aggregated


def print_metric_dict(
    d: Dict[str, float],
    title: Optional[str] = None,
    indent: int = 0,
):
    """Pretty-print a metric dictionary."""
    prefix = "  " * indent
    if title:
        print(f"{prefix}{title}")
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}  {k}:")
            for kk, vv in v.items():
                print(f"{prefix}    {kk}: {vv:.4f}")
        else:
            print(f"{prefix}  {k}: {v:.4f}")
