"""Model registry — classical ML models for PSC prediction"""

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


MODELS = {
    "M1_random_forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "M2_xgboost": lambda: _try_xgboost(),
    "M3_lightgbm": lambda: _try_lightgbm(),
    "M4_catboost": lambda: _try_catboost(),
    "M5_gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=5, random_state=42
    ),
    "M6_svr": lambda: SVR(kernel="rbf", C=10, epsilon=0.1),
    "M7_knn": lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    "M12_elastic_net": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    "M12_ridge": lambda: Ridge(alpha=1.0),
    "M12_lasso": lambda: Lasso(alpha=0.1, random_state=42),
}


def _try_xgboost():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)


def _try_lightgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)


def _try_catboost():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=0)


def get_model(model_id: str):
    factory = MODELS.get(model_id)
    if factory is None:
        raise ValueError(f"Unknown model: {model_id}")
    return factory()
