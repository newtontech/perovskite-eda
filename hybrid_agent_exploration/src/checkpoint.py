"""Checkpoint — 保存/加载探索进度，支持断点续跑"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_CHECKPOINT = "results/checkpoint.json"


def save_checkpoint(path: str, state: dict):
    state["timestamp"] = datetime.now().isoformat()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_checkpoint(path: str) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    with open(path) as f:
        return json.load(f)


def create_initial_state(total_runs: int) -> dict:
    return {
        "completed_runs": 0,
        "total_runs": total_runs,
        "results": [],
        "failed_runs": [],
        "best_r2": None,
        "best_pipeline": None,
    }


def update_state(state: dict, run_result: dict):
    state["completed_runs"] += 1
    state["results"].append(run_result)
    if run_result.get("status") == "success":
        r2 = run_result.get("metrics", {}).get("r2")
        if r2 is not None and (state["best_r2"] is None or r2 > state["best_r2"]):
            state["best_r2"] = r2
            state["best_pipeline"] = run_result.get("pipeline_config")
    else:
        state["failed_runs"].append(run_result)
