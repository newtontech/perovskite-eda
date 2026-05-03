"""Method Registry — 加载和管理方法注册表"""

import yaml
from pathlib import Path
from typing import Optional


def load_registry(path: str = "configs/method_registry.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_methods_by_layer(registry: dict, layer: int) -> dict:
    return {k: v for k, v in registry["methods"].items() if v["layer"] == layer}


def get_all_layers(registry: dict) -> dict[int, dict]:
    layers = {}
    for method_id, info in registry["methods"].items():
        layer = info["layer"]
        layers.setdefault(layer, {})[method_id] = info
    return dict(sorted(layers.items()))


def get_implemented_methods(registry: dict, layer: Optional[int] = None) -> dict:
    methods = registry["methods"]
    if layer is not None:
        methods = get_methods_by_layer(registry, layer)
    return {k: v for k, v in methods.items() if v.get("implemented", False)}


def compute_effective_weight(method: dict, current_year: int = 2026) -> float:
    base = method.get("weight", 1.0)
    year = method.get("novelty_year", 2010)
    age = current_year - year
    if age <= 1:
        novelty = 3.0
    elif age <= 3:
        novelty = 2.0
    else:
        novelty = 1.0
    availability = 1.5 if method.get("implemented", False) else 0.3
    psc = 1.2 if method.get("psc_verified", False) else 1.0
    return base * novelty * availability * psc
