"""CLI for bounded, resumable external evidence cache collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.evidence_cache_collector import (
    collect_evidence_cache,
    write_collection_report,
)
from harness.authenticity import CrossrefReferenceVerifier, PubChemMoleculeVerifier
from run_verified_discovery import default_cache_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements-csv",
        required=True,
        help="Preflight evidence_cache_requirements.csv.",
    )
    parser.add_argument(
        "--dataset-id", required=True, help="Stable dataset/run identifier."
    )
    parser.add_argument(
        "--cache-dir",
        help="Evidence cache directory; defaults to hybrid_agent_exploration/.cache/verified_discovery/<dataset-id>/evidence_cache.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        required=True,
        help="Maximum resolver calls for this batch.",
    )
    parser.add_argument(
        "--output-json", required=True, help="Collector report JSON path."
    )
    parser.add_argument(
        "--entity-type", choices=("all", "reference", "molecule"), default="all"
    )
    parser.add_argument("--retry-attempts", type=int, default=2)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress and refresh the output report every N attempted resolver calls. Use 0 to disable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--write-negative-cache",
        action="store_true",
        help="Persist resolver None results as negative cache entries. Default is false to avoid poisoning cache on swallowed transient failures.",
    )
    parser.add_argument(
        "--include-smiles",
        action="store_true",
        help="Attempt smiles:<SMILES> molecule keys. Default is false because the current PubChem resolver is CID-backed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else default_cache_dir(args.dataset_id)
    )

    def write_progress(snapshot: dict) -> None:
        print(
            "[evidence-cache-collector] progress "
            f"attempted={snapshot['attempted_count']}/{snapshot['planned_count']} "
            f"remaining_planned={snapshot['remaining_planned_count']} "
            f"positive={snapshot['positive_written_count']} "
            f"negative={snapshot['negative_written_count']} "
            f"no_evidence={snapshot['no_evidence_count']} "
            f"errors={snapshot['error_count']}",
            flush=True,
        )
        write_collection_report(snapshot, args.output_json)

    summary = collect_evidence_cache(
        requirements_csv=args.requirements_csv,
        cache_dir=cache_dir,
        dataset_id=args.dataset_id,
        max_requests=args.max_requests,
        entity_type=args.entity_type,
        retry_attempts=args.retry_attempts,
        dry_run=args.dry_run,
        include_smiles=args.include_smiles,
        write_negative_cache=args.write_negative_cache,
        progress_every=args.progress_every,
        progress_callback=write_progress if args.progress_every else None,
        reference_resolver=CrossrefReferenceVerifier(),
        molecule_resolver=PubChemMoleculeVerifier(),
    )
    report_path = write_collection_report(summary, args.output_json)
    print(f"[evidence-cache-collector] report={report_path}")
    print(f"[evidence-cache-collector] attempted={summary['attempted_count']}")
    print(
        f"[evidence-cache-collector] remaining_missing={summary['remaining_missing_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
