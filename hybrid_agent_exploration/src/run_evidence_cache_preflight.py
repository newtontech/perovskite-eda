"""CLI for planning external evidence-cache coverage.

This command is a read-only preflight: it inspects the source table and existing
``reference_cache.json`` / ``molecule_cache.json`` files without calling any
external service. Use it before a full ``external-cached`` research package run
to determine how many DOI and molecule identities still need cache collection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.evidence_cache_preflight import (
    summarize_evidence_cache_preflight,
    write_evidence_cache_preflight_artifacts,
)
from run_verified_discovery import default_cache_dir, load_input


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/XLSX source table to scan.")
    parser.add_argument("--dataset-id", required=True, help="Stable dataset/run identifier.")
    parser.add_argument("--source-name", help="Human-readable source table name; defaults to input file stem.")
    parser.add_argument(
        "--cache-dir",
        help="Evidence cache directory; defaults to hybrid_agent_exploration/.cache/verified_discovery/<dataset-id>/evidence_cache.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for preflight JSON/CSV/Markdown artifacts.")
    parser.add_argument("--max-rows", type=int, help="Optional input row cap for quick preflight smoke runs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir(args.dataset_id)
    df = load_input(input_path, max_rows=args.max_rows)
    artifacts = write_evidence_cache_preflight_artifacts(
        summarize_evidence_cache_preflight(
            df,
            dataset_id=args.dataset_id,
            source_name=args.source_name or input_path.stem,
            cache_dir=cache_dir,
            max_rows=args.max_rows,
        ),
        args.output_dir,
    )
    print(f"[evidence-cache-preflight] summary={artifacts.summary_json}")
    print(f"[evidence-cache-preflight] requirements={artifacts.requirements_csv}")
    print(f"[evidence-cache-preflight] report={artifacts.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
