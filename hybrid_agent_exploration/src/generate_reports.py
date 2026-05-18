#!/usr/bin/env python3
"""generate_reports.py

Batch-generate per-experiment reports and the master report from exploration results.

Usage:
    python src/generate_reports.py --input results/multi_agent_test --output results/reports
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reporting.per_experiment_report import PerExperimentReport
from reporting.master_report import MasterReport


def load_results(input_dir: Path) -> list[dict]:
    """Load all_results.json from an exploration run directory."""
    path = input_dir / "all_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_artifacts(agent_id: str, artifacts_dir: Path) -> dict | None:
    """Load saved artifacts for a given agent."""
    artifact_path = artifacts_dir / f"{agent_id}_artifacts.json"
    if not artifact_path.exists():
        return None
    with open(artifact_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate scientific reports from exploration results")
    parser.add_argument("--input", type=str, required=True, help="Directory containing all_results.json")
    parser.add_argument("--output", type=str, default="results/reports", help="Output directory for reports")
    parser.add_argument("--only-master", action="store_true", help="Only generate master report, skip per-experiment reports")
    parser.add_argument("--embed-images", action="store_true", help="Embed local images as base64 data URIs in generated markdown")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {input_dir}")
    results = load_results(input_dir)
    print(f"Loaded {len(results)} experiment results")

    artifacts_dir = PROJECT_ROOT / "results" / "report_artifacts"

    # Generate per-experiment reports
    if not args.only_master:
        successful = [r for r in results if r.get("status") == "success"]
        print(f"Generating per-experiment reports for {len(successful)} successful experiments...")
        for i, result in enumerate(successful, 1):
            agent_id = result.get("agent_id", f"agent_{i:03d}")
            artifacts = load_artifacts(agent_id, artifacts_dir)
            report = PerExperimentReport(
                result=result,
                artifacts=artifacts,
                output_dir=output_dir / "per_experiment",
                embed_images=args.embed_images,
            )
            report_path = report.generate()
            print(f"  [{i}/{len(successful)}] {agent_id} -> {report_path}")
        print("Per-experiment reports complete.")

    # Generate master report
    print("Generating master report...")
    master = MasterReport(results, output_dir=output_dir, embed_images=args.embed_images)
    master_path = master.generate()
    print(f"Master report -> {master_path}")

    print("\nAll reports generated successfully.")


if __name__ == "__main__":
    main()
