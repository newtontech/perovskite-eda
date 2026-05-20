"""Weighted Sampler — 加权随机采样器，像摇骰子一样选择方法"""

import numpy as np
from typing import Optional

from .registry import get_methods_by_layer, compute_effective_weight


class WeightedSampler:
    def __init__(self, registry: dict, seed: Optional[int] = None, only_implemented: bool = True):
        self.registry = registry
        self.only_implemented = only_implemented
        self.rng = np.random.default_rng(seed)

    def sample_layer(self, layer: int, n: int = 1) -> list[str]:
        methods = get_methods_by_layer(self.registry, layer)
        if self.only_implemented:
            methods = {k: v for k, v in methods.items() if v.get("implemented", False)}
        if not methods:
            return []
        keys = list(methods.keys())
        weights = np.array([compute_effective_weight(v) for v in methods.values()])
        weights = weights / weights.sum()
        chosen = self.rng.choice(keys, size=n, replace=False, p=weights)
        return list(chosen)

    def sample_pipeline(self) -> dict:
        pipeline = {}
        for layer in [1, 2, 3, 4, 5]:
            chosen = self.sample_layer(layer, n=1)
            pipeline[layer] = chosen[0] if chosen else None
        return pipeline

    def get_weight_distribution(self, layer: int) -> dict[str, float]:
        methods = get_methods_by_layer(self.registry, layer)
        dist = {}
        for k, v in methods.items():
            dist[k] = compute_effective_weight(v)
        total = sum(dist.values())
        return {k: round(v / total, 4) for k, v in dist.items()}
