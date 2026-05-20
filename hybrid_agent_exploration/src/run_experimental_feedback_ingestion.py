"""Validate and package a user-supplied closed-loop feedback CSV."""

from __future__ import annotations

import argparse
import json

from closed_loop.feedback_ingestion import ingest_experimental_feedback


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-dir", required=True, help="closed_loop/iteration_XXX directory.")
    parser.add_argument("--feedback-csv", required=True, help="User-supplied experimental feedback CSV.")
    parser.add_argument("--output-dir", help="Optional output dir. Defaults to iteration_dir/experimental_feedback.")
    parser.add_argument(
        "--feedback-data-scope",
        default="external_lab",
        help="Scope label for feedback provenance, for example external_lab or unit_test_fixture.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = ingest_experimental_feedback(
        iteration_dir=args.iteration_dir,
        feedback_csv=args.feedback_csv,
        output_dir=args.output_dir,
        feedback_data_scope=args.feedback_data_scope,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output_dir": str(artifacts.output_dir),
                "validated_feedback_csv": str(artifacts.validated_feedback_csv),
                "feedback_vs_prediction_csv": str(artifacts.feedback_vs_prediction_csv),
                "feedback_ingestion_manifest_json": str(artifacts.feedback_ingestion_manifest_json),
                "feedback_validation_report_json": str(artifacts.feedback_validation_report_json),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
