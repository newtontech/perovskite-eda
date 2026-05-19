"""Report — 汇总探索结果，输出报告"""

import json
from pathlib import Path
from datetime import datetime


def collect_results(results_dir: str = "results/exploration_logs") -> list[dict]:
    results = []
    p = Path(results_dir)
    if not p.exists():
        return results
    for f in sorted(p.glob("run_*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def generate_report(results: list[dict], output_path: str = "results/comparison_reports/final_report.md"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "error"]
    successful.sort(key=lambda r: r.get("metrics", {}).get("r2", float("-inf")), reverse=True)

    lines = [
        f"# Exploration Report",
        f"Generated: {datetime.now().isoformat()}",
        f"",
        f"## Summary",
        f"- Total runs: {len(results)}",
        f"- Successful: {len(successful)}",
        f"- Failed: {len(failed)}",
        f"- Best R²: {successful[0]['metrics']['r2']:.4f}" if successful else "- Best R²: N/A",
        f"",
        f"## Top 10 Pipelines",
        f"",
        f"| Rank | R² | RMSE | Data | Feature | Model | Eval | Duration |",
        f"|------|-----|------|------|---------|-------|------|----------|",
    ]

    for i, r in enumerate(successful[:10]):
        cfg = r["pipeline_config"]
        m = r["metrics"]
        lines.append(
            f"| {i+1} | {m.get('r2', 'N/A'):.4f} | {m.get('rmse', 'N/A'):.4f} "
            f"| {cfg.get(1, '?')} | {cfg.get(2, '?')} | {cfg.get(3, '?')} "
            f"| {cfg.get(4, '?')} | {r['duration_sec']}s |"
        )

    if failed:
        lines.extend(["", "## Failed Runs", ""])
        for r in failed[:10]:
            cfg = r["pipeline_config"]
            lines.append(f"- `{cfg.get(2, '?')} + {cfg.get(3, '?')}`: {r.get('error', 'unknown')}")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    return report


def print_summary(results: list[dict]):
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "error"]
    print(f"\n{'='*60}")
    print(f"EXPLORATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(results)} | Success: {len(successful)} | Failed: {len(failed)}")
    if successful:
        best = max(successful, key=lambda r: r["metrics"].get("r2", float("-inf")))
        cfg = best["pipeline_config"]
        print(f"\nBest R²: {best['metrics']['r2']:.4f}")
        print(f"  Data:      {cfg.get(1)}")
        print(f"  Feature:   {cfg.get(2)}")
        print(f"  Model:     {cfg.get(3)}")
        print(f"  Eval:      {cfg.get(4)}")
        print(f"  Duration:  {best['duration_sec']}s")
    print(f"{'='*60}")
