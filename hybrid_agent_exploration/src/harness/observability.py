"""observability.py — Structured logging and metrics collection."""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class Observability:
    """Capture agent execution traces as JSONL for post-hoc analysis.

    Writes are protected by an in-process lock.  When the same log file is
    shared across *spawn* processes, the caller should ensure each process
    receives a distinct file path (the default timestamp-based name achieves
    this when instances are created independently).
    """

    def __init__(self, log_dir: Path | str = "results/harness_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = (
            self.log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_agents.jsonl"
        )
        self._records: list[dict] = []
        self._lock = threading.Lock()

    def _emit(self, event_type: str, agent_id: str | None, data: dict):
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "agent_id": agent_id,
            "data": data,
        }
        with self._lock:
            self._records.append(record)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def log_agent_start(self, agent_id: str, config: dict):
        self._emit("agent_start", agent_id, {"config": config})

    def log_agent_end(self, agent_id: str, metrics: dict, duration_sec: float):
        self._emit(
            "agent_end", agent_id, {"metrics": metrics, "duration_sec": duration_sec}
        )

    def log_layer_timing(self, agent_id: str, layer: str, duration_sec: float):
        self._emit(
            "layer_timing", agent_id, {"layer": layer, "duration_sec": duration_sec}
        )

    def log_error(self, agent_id: str, error: str, traceback_str: str | None = None):
        self._emit("error", agent_id, {"error": error, "traceback": traceback_str})

    def get_summary(self) -> dict[str, Any]:
        """Aggregate metrics across all recorded agents."""
        with self._lock:
            records = list(self._records)

        starts = [r for r in records if r["event_type"] == "agent_start"]
        ends = [r for r in records if r["event_type"] == "agent_end"]
        errors = [r for r in records if r["event_type"] == "error"]
        timings = [r for r in records if r["event_type"] == "layer_timing"]

        total = len(starts)
        completed = len(ends)
        failed = len(errors)
        total_duration = sum(e["data"].get("duration_sec", 0) for e in ends)

        layer_times: dict[str, list[float]] = {}
        for t in timings:
            layer = t["data"]["layer"]
            layer_times.setdefault(layer, []).append(t["data"]["duration_sec"])

        layer_summary = {
            layer: {"mean": sum(v) / len(v), "count": len(v)}
            for layer, v in layer_times.items()
        }

        return {
            "total_agents": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total else 0,
            "total_duration_sec": total_duration,
            "mean_duration_sec": total_duration / completed if completed else 0,
            "layer_summary": layer_summary,
            "log_file": str(self._log_path),
        }
