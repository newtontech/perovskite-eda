"""worker_agent.py

Single WorkerAgent: receives a pipeline config, executes the full cross-layer
pipeline, and returns a structured result dict.
"""

import warnings
import traceback
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_worker(agent_id: str, config: dict) -> Dict[str, Any]:
    """Execute one full pipeline for a single Agent.
    
    Parameters
    ----------
    agent_id : str
        Unique identifier for this worker.
    config : dict
        Cross-layer pipeline config from cross_layer_sampler.
    
    Returns
    -------
    dict
        Structured result with status, metrics, timing, and error info.
    """
    start = time.time()
    result = {
        "agent_id": agent_id,
        "config": config,
        "status": "unknown",
        "metrics": {},
        "n_samples": 0,
        "n_features": 0,
        "duration_sec": 0.0,
        "error": None,
        "traceback": None,
    }
    
    try:
        # Import pipeline runner
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from pipeline import run_pipeline
        from cross_layer_sampler import format_config_for_pipeline
        
        # Convert config to pipeline format and run
        pipeline_cfg = format_config_for_pipeline(config)
        pipeline_result = run_pipeline(pipeline_cfg)
        
        # Copy results
        result["status"] = pipeline_result["status"]
        result["metrics"] = pipeline_result.get("metrics", {})
        result["n_samples"] = pipeline_result.get("n_samples", 0)
        result["n_features"] = pipeline_result.get("n_features", 0)
        
        if pipeline_result["status"] == "error":
            result["error"] = pipeline_result.get("error", "Unknown error")
            result["traceback"] = pipeline_result.get("traceback", "")
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["traceback"] = traceback.format_exc()
    
    result["duration_sec"] = round(time.time() - start, 2)
    return result


def run_worker_star(args):
    """Wrapper for multiprocessing.Pool.map that unpacks (agent_id, config) tuple."""
    agent_id, config = args
    return run_worker(agent_id, config)


if __name__ == "__main__":
    # Quick smoke test
    test_config = {
        "layer1": {"method_id": "agentic_veryloose", "strategy": "agentic_veryloose"},
        "layer2": {"method_id": "F21_rdkit_basic"},
        "layer3": {"method_id": "M31_random_forest"},
        "layer4": {"method_id": "E43_5fold_cv"},
        "layer5": {"method_id": "D54_report_only"},
        "target": "delta_pce",
        "baseline_as_feature": True,
        "_hash": "test123",
    }
    res = run_worker("test_agent", test_config)
    print("Status:", res["status"])
    print("Metrics:", res["metrics"])
    print("Duration:", res["duration_sec"], "s")
