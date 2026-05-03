"""Pipeline — 组装和执行完整的5层管线

扩展版本：支持 Layer 1 策略选择和 Layer 5 筛选
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from registry import load_registry
except ImportError:
    from .registry import load_registry

# Layer 1 cleaning strategies
L1_STRATEGIES = {
    "agentic_veryloose": {
        "smiles_validity": "NonEmpty",
        "baseline_pce_bounds": "Physical_Only",
        "treated_pce_bounds": "Physical_Only",
        "delta_pce_bounds": "AllObserved",
        "jv_consistency": "PCE_Only",
        "device_info_completeness": "NoDeviceFilter",
        "literature_quality": "NoLiteratureFilter",
        "deduplication": "NoDeduplication",
        "descriptor_bounds": "NoFilter",
    },
    "agentic_standard": {
        "smiles_validity": "RDKit_Strict",
        "baseline_pce_bounds": "Standard_Devices",
        "treated_pce_bounds": "Standard_Treated",
        "delta_pce_bounds": "ReasonableRange",
        "jv_consistency": "FullJV_Standard",
        "device_info_completeness": "StructureAndComposition",
        "literature_quality": "HasPublicationInfo",
        "deduplication": "Unique_Molecule_Baseline",
        "descriptor_bounds": "DrugLike_Standard",
    },
    "agentic_strict": {
        "smiles_validity": "RDKit_Strict",
        "baseline_pce_bounds": "HighQuality_Devices",
        "treated_pce_bounds": "HighQuality_Treated",
        "delta_pce_bounds": "ReasonableRange",
        "jv_consistency": "FullJV_Strict",
        "device_info_completeness": "FullDeviceInfo",
        "literature_quality": "PeerReviewed_Only",
        "deduplication": "Unique_Device",
        "descriptor_bounds": "DrugLike_Strict",
    },
    "traditional": "traditional",  # special marker
}


def load_data(config: dict, registry: dict = None) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load data with optional Layer 1 cleaning strategy.
    
    config keys:
      - layer1: dict with 'strategy' key (agentic_veryloose / agentic_standard / agentic_strict / traditional)
      - or legacy: method_id at key 1 for registry lookup
    """
    # Determine Layer 1 strategy
    l1_config = config.get("layer1", {})
    strategy_name = l1_config.get("strategy", "agentic_veryloose")
    
    # Default data path
    data_path = "/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx"
    cache_dir = Path(__file__).resolve().parents[1] / "explorations" / "data_source_exploration" / ".cache"
    cache_csv = cache_dir / "merged_llm_crossref_data_streaming_with_chemical_data_fast.csv"
    
    # Load from cache CSV if available (much faster than Excel)
    if cache_csv.exists():
        df = pd.read_csv(cache_csv, low_memory=False)
    else:
        df = pd.read_excel(data_path)
    
    # Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    smiles_col = "smiles"
    pce = "jv_reverse_scan_pce"
    pce_wo = "jv_reverse_scan_pce_without_modulator"
    
    # Compute delta_pce if not present
    if "delta_pce" not in df.columns and pce in df.columns and pce_wo in df.columns:
        df[pce] = pd.to_numeric(df[pce], errors="coerce")
        df[pce_wo] = pd.to_numeric(df[pce_wo], errors="coerce")
        df["delta_pce"] = df[pce] - df[pce_wo]
    
    # Apply Layer 1 cleaning strategy
    if strategy_name == "traditional":
        # Use traditional cleaning from explore_data_sources
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorations" / "data_source_exploration"))
        from explore_data_sources import clean_psc_data
        df = clean_psc_data(df)
    elif strategy_name in L1_STRATEGIES and strategy_name != "traditional":
        # Use agentic cleaning
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
        from agentic_data_cleaner import execute_strategy
        import yaml
        action_space_path = Path(__file__).resolve().parents[1] / "configs" / "cleaning_action_space.yaml"
        with open(action_space_path) as f:
            action_space = yaml.safe_load(f)
        strategy = L1_STRATEGIES[strategy_name]
        df = execute_strategy(df, strategy, action_space)
    else:
        # Minimal default: just drop NaN SMILES and clamp PCE
        df = df.dropna(subset=[smiles_col])
        if pce in df.columns:
            df[pce] = pd.to_numeric(df[pce], errors="coerce").clip(0, 30)
        if pce_wo in df.columns:
            df[pce_wo] = pd.to_numeric(df[pce_wo], errors="coerce").clip(0, 30)
    
    # Ensure target exists
    target = config.get("target", "delta_pce")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found after cleaning. Available: {list(df.columns)}")
    
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[smiles_col, target])
    
    return df, df[target], smiles_col


def compute_features(df: pd.DataFrame, smiles_col: str, method_id: str) -> np.ndarray:
    # Ensure project root is on path for features/ imports
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    if method_id == "F21_rdkit_basic":
        from features.rdkit_descriptors import compute_basic_descriptors
        feat_df = compute_basic_descriptors(df[smiles_col])
        return feat_df.fillna(0).values

    if method_id in ("F22_ecfp4", "F22_ecfp6", "F22_maccs", "F22_krfp",
                      "F22_atom_pair", "F22_topological_torsion"):
        from features.fingerprints import get_fingerprint
        mapping = {
            "F22_ecfp4": "F2_ecfp", "F22_ecfp6": "F2_ecfp6",
            "F22_maccs": "F3_maccs", "F22_krfp": "F4_krfp",
            "F22_atom_pair": "F5_atom_pair", "F22_topological_torsion": "F6_topological_torsion",
        }
        return get_fingerprint(mapping[method_id], df[smiles_col])

    raise ValueError(f"Feature method not implemented: {method_id}")


def get_model(method_id: str):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from models.model_registry import get_model as _get_model
    mapping = {
        "M31_random_forest": "M1_random_forest",
        "M31_xgboost": "M2_xgboost",
        "M31_lightgbm": "M3_lightgbm",
        "M31_catboost": "M4_catboost",
        "M31_gradient_boosting": "M5_gradient_boosting",
        "M31_svr": "M6_svr",
        "M31_knn": "M7_knn",
        "M31_elastic_net": "M12_elastic_net",
        "M31_ridge": "M12_ridge",
        "M31_lasso": "M12_lasso",
    }
    sk_id = mapping.get(method_id)
    if sk_id:
        return _get_model(sk_id)
    raise ValueError(f"Model not implemented: {method_id}")


def evaluate(model, X: np.ndarray, y: np.ndarray, method_id: str, n_splits: int = 5) -> dict:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if method_id in ("E43_5fold_cv", "E43_10fold_cv"):
        n_splits = 5 if method_id == "E43_5fold_cv" else 10
        scores = cross_val_score(model, Xs, y, cv=n_splits, scoring="r2")
        return {"r2": float(scores.mean()), "r2_std": float(scores.std()), "strategy": method_id}

    if method_id == "E42_random_split":
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        return {"r2": float(r2), "rmse": rmse, "strategy": method_id}

    if method_id == "E44_optuna":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return _optuna_tune(model, Xs, y, n_trials=50)

    if method_id == "E44_optuna_large":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return _optuna_tune(model, Xs, y, n_trials=200)

    if method_id == "E44_grid_search":
        from sklearn.model_selection import GridSearchCV
        param_grid = _get_param_grid(model)
        gs = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1)
        gs.fit(Xs, y)
        return {"r2": float(gs.best_score_), "best_params": str(gs.best_params_), "strategy": method_id}

    if method_id == "E45_shap":
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        try:
            import shap
            explainer = shap.Explainer(model, X_train)
            sv = explainer(X_test[:50])
            top_feat = int(np.argmax(np.mean(np.abs(sv.values), axis=0)))
        except Exception:
            top_feat = -1
        return {"r2": float(r2), "rmse": rmse, "top_shap_feature": top_feat, "strategy": method_id}

    # Fallback: default 5-fold CV
    scores = cross_val_score(model, Xs, y, cv=n_splits, scoring="r2")
    return {"r2": float(scores.mean()), "r2_std": float(scores.std()), "strategy": method_id}


def run_screening(model, X_train: np.ndarray, y_train: np.ndarray, 
                  X_candidates: np.ndarray, method_id: str, k: int = 20) -> dict:
    """Layer 5 — run virtual screening or closed-loop simulation."""
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_cand = scaler.transform(X_candidates)
    model.fit(Xs_train, y_train)
    
    if method_id == "D53_top_k":
        preds = model.predict(Xs_cand)
        top_idx = np.argsort(preds)[::-1][:k]
        return {"top_k_indices": top_idx.tolist(), "top_k_scores": preds[top_idx].tolist(), "strategy": method_id}
    
    if method_id == "D54_report_only":
        return {"strategy": method_id, "note": "Model trained and reported."}
    
    return {"strategy": method_id, "note": "Layer 5 method not fully implemented"}


def _optuna_tune(model, X, y, n_trials=50):
    import optuna
    from sklearn.model_selection import cross_val_score
    model_name = type(model).__name__

    def objective(trial):
        params = _sample_params(trial, model_name)
        m = model.__class__(**params)
        scores = cross_val_score(m, X, y, cv=3, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"r2": float(study.best_value), "best_params": str(study.best_params), "strategy": f"optuna_{n_trials}"}


def _sample_params(trial, model_name):
    if "RandomForest" in model_name:
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 500), "max_depth": trial.suggest_int("max_depth", 3, 20), "random_state": 42, "n_jobs": -1}
    if "XGB" in model_name:
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 500), "max_depth": trial.suggest_int("max_depth", 3, 15), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True), "random_state": 42, "n_jobs": -1}
    if "LGBM" in model_name:
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 500), "max_depth": trial.suggest_int("max_depth", 3, 15), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True), "random_state": 42, "n_jobs": -1, "verbose": -1}
    if "CatBoost" in model_name:
        return {"iterations": trial.suggest_int("iterations", 50, 500), "depth": trial.suggest_int("depth", 3, 10), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True), "random_state": 42, "verbose": 0}
    if "SVR" in model_name:
        return {"C": trial.suggest_float("C", 0.1, 100, log=True), "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True), "kernel": "rbf"}
    if "KNN" in model_name:
        return {"n_neighbors": trial.suggest_int("n_neighbors", 1, 20), "weights": trial.suggest_categorical("weights", ["uniform", "distance"]), "n_jobs": -1}
    if "ElasticNet" in model_name:
        return {"alpha": trial.suggest_float("alpha", 0.001, 10, log=True), "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0), "random_state": 42}
    if "Ridge" in model_name:
        return {"alpha": trial.suggest_float("alpha", 0.01, 100, log=True)}
    if "Lasso" in model_name:
        return {"alpha": trial.suggest_float("alpha", 0.001, 10, log=True), "random_state": 42}
    if "GradientBoosting" in model_name:
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 500), "max_depth": trial.suggest_int("max_depth", 2, 10), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True), "random_state": 42}
    return {}


def _get_param_grid(model):
    name = type(model).__name__
    if "RandomForest" in name:
        return {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
    if "GradientBoosting" in name:
        return {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
    if "SVR" in name:
        return {"C": [1, 10, 100], "epsilon": [0.01, 0.1]}
    if "KNN" in name:
        return {"n_neighbors": [3, 5, 10]}
    return {}


def run_pipeline(pipeline_config: dict, registry: dict = None) -> dict:
    """Execute a full cross-layer pipeline.
    
    Supports both legacy integer-key config and new string-key config:
      legacy: {1: "S11_literature", 2: "F21_rdkit_basic", 3: "M31_rf", 4: "E43_5fold_cv"}
      new:    {"layer1": {...}, "layer2": {...}, "layer3": {...}, "layer4": {...}, "layer5": {...}}
    """
    cfg = {}
    for k, v in pipeline_config.items():
        cfg[int(k) if str(k).isdigit() else k] = v

    start = time.time()
    result = {
        "pipeline_config": pipeline_config,
        "status": "unknown",
        "metrics": {},
        "duration_sec": 0,
        "error": None,
    }
    try:
        # Layer 1: Data loading + cleaning
        df, y, smiles_col = load_data(cfg, registry)
        
        # Layer 2: Feature generation
        l2_method = cfg.get(2) or cfg.get("layer2", {}).get("method_id", "F21_rdkit_basic")
        X = compute_features(df, smiles_col, l2_method)
        
        # Optionally append baseline PCE as feature
        baseline_as_feature = cfg.get("baseline_as_feature", False)
        if baseline_as_feature and "jv_reverse_scan_pce_without_modulator" in df.columns:
            baseline = pd.to_numeric(df["jv_reverse_scan_pce_without_modulator"], errors="coerce").fillna(0).values.reshape(-1, 1)
            X = np.hstack([X, baseline])
        
        # Layer 3: Model
        l3_method = cfg.get(3) or cfg.get("layer3", {}).get("method_id", "M31_random_forest")
        model = get_model(l3_method)
        
        # Layer 4: Evaluation
        l4_method = cfg.get(4) or cfg.get("layer4", {}).get("method_id", "E43_5fold_cv")
        metrics = evaluate(model, X, y.values if hasattr(y, "values") else y, l4_method)
        
        # Layer 5: Screening (optional)
        l5_method = cfg.get(5) or cfg.get("layer5", {}).get("method_id")
        if l5_method and l5_method not in ("D54_report_only", None):
            # Simple placeholder: use a subset as "candidates"
            n_cand = min(100, len(X))
            X_cand = X[:n_cand]
            y_cand = y.values[:n_cand] if hasattr(y, "values") else y[:n_cand]
            screen_result = run_screening(model, X, y.values if hasattr(y, "values") else y, X_cand, l5_method)
            metrics["layer5"] = screen_result
        
        result["status"] = "success"
        result["metrics"] = metrics
        result["n_samples"] = int(len(df))
        result["n_features"] = int(X.shape[1])
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["traceback"] = traceback.format_exc()
    result["duration_sec"] = round(time.time() - start, 2)
    return result
