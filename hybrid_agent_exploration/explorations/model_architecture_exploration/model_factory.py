"""
model_factory.py
Unified model registry / wrapper for Layer 3 — M31 Classical ML models.

Supported models (M31):
  - Random Forest
  - XGBoost
  - LightGBM
  - SVR
  - KNN

Usage:
    factory = ModelFactory()
    model = factory.create("xgboost", n_estimators=200, max_depth=6)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


class BaseModelWrapper:
    """Thin wrapper exposing a uniform fit / predict API."""

    def __init__(self, estimator: Any, name: str):
        self.estimator = estimator
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModelWrapper":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Return feature importances if the underlying model supports it."""
        if hasattr(self.estimator, "feature_importances_"):
            return self.estimator.feature_importances_
        return None


class ModelFactory:
    """Registry for classical ML regressors (M31)."""

    _registry: Dict[str, Dict[str, Any]] = {
        "random_forest": {
            "class": RandomForestRegressor,
            "defaults": {
                "n_estimators": 100,
                "max_depth": 12,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        "xgboost": {
            "class": None,  # resolved at runtime
            "defaults": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": 1,
                "verbosity": 0,
            },
        },
        "lightgbm": {
            "class": None,  # resolved at runtime
            "defaults": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": 1,
                "verbosity": -1,
            },
        },
        "svr": {
            "class": SVR,
            "defaults": {
                "kernel": "rbf",
                "C": 1.0,
                "epsilon": 0.1,
                "gamma": "scale",
            },
        },
        "knn": {
            "class": KNeighborsRegressor,
            "defaults": {
                "n_neighbors": 5,
                "weights": "distance",
                "n_jobs": -1,
            },
        },
    }

    def __init__(self):
        # Runtime resolution for optional packages
        if xgb is not None:
            self._registry["xgboost"]["class"] = xgb.XGBRegressor
        if lgb is not None:
            self._registry["lightgbm"]["class"] = lgb.LGBMRegressor

    def list_models(self) -> list[str]:
        """Return available model names."""
        available = []
        for name, meta in self._registry.items():
            if meta["class"] is not None:
                available.append(name)
        return available

    def create(self, name: str, **overrides: Any) -> BaseModelWrapper:
        """
        Instantiate a model by name.

        Parameters
        ----------
        name : str
            One of "random_forest", "xgboost", "lightgbm", "svr", "knn".
        **overrides :
            Keyword arguments that override the default hyper-parameters.
        """
        name = name.lower().replace("-", "_")
        if name not in self._registry:
            raise ValueError(
                f"Unknown model '{name}'. Available: {self.list_models()}"
            )

        meta = self._registry[name]
        cls = meta["class"]
        if cls is None:
            raise ImportError(
                f"Model '{name}' is not available because its package is not installed."
            )

        params = {**meta["defaults"], **overrides}
        estimator = cls(**params)
        return BaseModelWrapper(estimator, name)

    def get_defaults(self, name: str) -> Dict[str, Any]:
        """Return default hyper-parameters for a model."""
        name = name.lower().replace("-", "_")
        return dict(self._registry[name]["defaults"])
