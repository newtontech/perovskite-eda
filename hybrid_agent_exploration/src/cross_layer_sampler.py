"""cross_layer_sampler.py

Generate random cross-layer pipeline configurations for multi-Agent exploration.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Load registries
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "method_registry.yaml"
ACTION_SPACE_PATH = PROJECT_ROOT / "configs" / "cleaning_action_space.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _effective_weight(method: dict) -> float:
    """Compute effective weight for sampling bias."""
    base = method.get("weight", 1.0)
    year = method.get("novelty_year", 2010)
    age = 2026 - year
    if age <= 1:
        novelty = 3.0
    elif age <= 3:
        novelty = 2.0
    else:
        novelty = 1.0
    availability = 1.5 if method.get("implemented", False) else 0.3
    psc = 1.2 if method.get("psc_verified", False) else 1.0
    return base * novelty * availability * psc


def get_implemented_methods_by_layer(registry: dict, layer: int) -> Dict[str, dict]:
    """Return all implemented methods for a given layer."""
    return {
        k: v for k, v in registry["methods"].items()
        if v.get("layer") == layer and v.get("implemented", False)
    }


# Layer 1 strategies (from agentic cleaning action space)
LAYER1_STRATEGIES = [
    {"name": "agentic_veryloose", "strategy": "agentic_veryloose"},
    {"name": "agentic_standard", "strategy": "agentic_standard"},
    {"name": "agentic_strict", "strategy": "agentic_strict"},
    {"name": "traditional", "strategy": "traditional"},
]


def _config_hash(config: dict) -> str:
    """Deterministic hash for deduplication."""
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def sample_layer_choice(methods: Dict[str, dict], weighted: bool = True) -> str:
    """Sample one method ID from a layer's method dict."""
    ids = list(methods.keys())
    if not ids:
        raise ValueError("No implemented methods available")
    if weighted:
        weights = [_effective_weight(methods[mid]) for mid in ids]
        return random.choices(ids, weights=weights, k=1)[0]
    return random.choice(ids)


# Methods that are too slow for multi-agent parallel exploration (fast_only mode)
SLOW_EVAL_METHODS = {"E44_optuna", "E44_optuna_large", "E44_grid_search"}
SLOW_FEATURE_METHODS = {"F22_krfp"}  # 4860-bit KRFP is very slow


def generate_pipeline_config(
    registry: dict,
    target: str = "delta_pce",
    baseline_as_feature: bool = True,
    weighted: bool = True,
    fast_only: bool = True,
) -> dict:
    """Generate one random cross-layer pipeline configuration."""
    # Layer 1: data cleaning strategy
    l1 = random.choice(LAYER1_STRATEGIES)
    
    # Layer 2: features
    l2_methods = get_implemented_methods_by_layer(registry, 2)
    if fast_only:
        l2_methods = {k: v for k, v in l2_methods.items() if k not in SLOW_FEATURE_METHODS}
    l2_id = sample_layer_choice(l2_methods, weighted)
    
    # Layer 3: model
    l3_methods = get_implemented_methods_by_layer(registry, 3)
    l3_id = sample_layer_choice(l3_methods, weighted)
    
    # Layer 4: evaluation
    l4_methods = get_implemented_methods_by_layer(registry, 4)
    if fast_only:
        l4_methods = {k: v for k, v in l4_methods.items() if k not in SLOW_EVAL_METHODS}
    l4_id = sample_layer_choice(l4_methods, weighted)
    
    # Layer 5: screening
    l5_methods = get_implemented_methods_by_layer(registry, 5)
    if l5_methods:
        l5_id = sample_layer_choice(l5_methods, weighted)
    else:
        l5_id = "D54_report_only"
    
    config = {
        "layer1": {"method_id": l1["name"], "strategy": l1["strategy"]},
        "layer2": {**l2_methods.get(l2_id, {}).get("params", {}), "method_id": l2_id},
        "layer3": {**l3_methods.get(l3_id, {}).get("params", {}), "method_id": l3_id},
        "layer4": {**l4_methods.get(l4_id, {}).get("params", {}), "method_id": l4_id},
        "layer5": {**l5_methods.get(l5_id, {}).get("params", {}), "method_id": l5_id},
        "target": target,
        "baseline_as_feature": baseline_as_feature,
    }
    config["_hash"] = _config_hash(config)
    return config


def generate_unique_configs(
    n: int,
    registry: dict = None,
    target: str = "delta_pce",
    baseline_as_feature: bool = True,
    weighted: bool = True,
    fast_only: bool = True,
    max_attempts: int = 1000,
) -> List[dict]:
    """Generate N unique cross-layer pipeline configurations."""
    if registry is None:
        registry = _load_yaml(REGISTRY_PATH)
    
    seen_hashes = set()
    configs = []
    attempts = 0
    
    while len(configs) < n and attempts < max_attempts:
        cfg = generate_pipeline_config(registry, target, baseline_as_feature, weighted, fast_only)
        h = cfg["_hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            configs.append(cfg)
        attempts += 1
    
    return configs


def format_config_for_pipeline(config: dict) -> dict:
    """Convert sampler config format to pipeline.py format."""
    return {
        "layer1": config["layer1"],
        "layer2": config["layer2"],
        "layer3": config["layer3"],
        "layer4": config["layer4"],
        "layer5": config["layer5"],
        "target": config["target"],
        "baseline_as_feature": config["baseline_as_feature"],
    }


if __name__ == "__main__":
    # Quick test
    registry = _load_yaml(REGISTRY_PATH)
    configs = generate_unique_configs(5, registry)
    for i, cfg in enumerate(configs):
        print(f"\n--- Config {i+1} ---")
        print(f"  L1: {cfg['layer1']['method_id']}")
        print(f"  L2: {cfg['layer2']['method_id']}")
        print(f"  L3: {cfg['layer3']['method_id']}")
        print(f"  L4: {cfg['layer4']['method_id']}")
        print(f"  L5: {cfg['layer5']['method_id']}")
        print(f"  Target: {cfg['target']} | Baseline: {cfg['baseline_as_feature']}")
