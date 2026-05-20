"""CLI for merging evidence cache shard JSON into the canonical cache."""

from __future__ import annotations

import argparse

from data.evidence_cache_collector import (
    merge_evidence_cache_shards,
    write_collection_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-cache-dir",
        required=True,
        help="Canonical evidence cache directory to update atomically.",
    )
    parser.add_argument(
        "--shard-cache-dir",
        action="append",
        required=True,
        help="Shard evidence cache directory. Repeat for multiple shards.",
    )
    parser.add_argument(
        "--entity-type",
        choices=("all", "reference", "molecule"),
        default="all",
        help="Cache file type to merge. Use reference for DOI-only shard batches.",
    )
    parser.add_argument(
        "--conflict-policy",
        choices=("fail", "keep-canonical"),
        default="fail",
        help="How to handle conflicting cache entries.",
    )
    parser.add_argument("--output-json", required=True, help="Merge report JSON path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = merge_evidence_cache_shards(
        canonical_cache_dir=args.canonical_cache_dir,
        shard_cache_dirs=args.shard_cache_dir,
        entity_type=args.entity_type,
        conflict_policy=args.conflict_policy,
    )
    report_path = write_collection_report(summary, args.output_json)
    print(f"[evidence-cache-merge] report={report_path}")
    print(f"[evidence-cache-merge] merged={summary['merged_counts']}")
    print(f"[evidence-cache-merge] conflicts={summary['conflict_counts']}")
    return 1 if summary.get("merge_status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
