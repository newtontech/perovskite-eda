"""Run the verified PSC candidate-discovery research package.

This CLI connects the verified dataset/discovery workflow, optional external
candidate-library normalization, report/SI generation, and root provenance
manifest into one reproducible artifact directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reporting.root_provenance_manifest import generate_root_provenance_manifest
from reporting.si_generator import SIGenerator
from reporting.top_journal_report import TopJournalReport
from run_verified_discovery import build_authenticator, default_cache_dir, load_input
from screening.candidate_library_builder import CandidateLibraryBuilder
from screening.verified_candidate_discovery import CANDIDATE_LIBRARY_CONTRACT_VERSION
from screening.verified_discovery_workflow import VerifiedDiscoveryWorkflow


@dataclass(frozen=True)
class ResearchPackageArtifacts:
    """Top-level paths emitted by a research package run."""

    output_dir: Path
    verified_discovery_dir: Path
    candidate_library_dir: Path | None
    report_dir: Path
    root_provenance_manifest_json: Path
    package_manifest_json: Path


def run_research_package(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    evidence_mode: str = "external-cached",
    candidate_source_path: str | Path | None = None,
    candidate_source_name: str | None = None,
    cache_dir: str | Path | None = None,
    max_rows: int | None = None,
    min_verified_rows: int = 10,
    top_k: int = 100,
    report_quality_target: str = "top-journal",
    verified_discovery_top_n: int | None = None,
) -> ResearchPackageArtifacts:
    """Generate the full verified-discovery research artifact package."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    verified_discovery_dir = output_root / "verified_discovery"
    report_dir = output_root / "report_bundle"

    df = load_input(input_path, max_rows=max_rows)
    cache_root = Path(cache_dir) if cache_dir else default_cache_dir(dataset_id)
    authenticator = build_authenticator(evidence_mode, df, cache_root)

    candidate_library_dir: Path | None = None
    candidate_pool: Path | None = None
    candidate_library_rows: int | None = None
    if candidate_source_path is not None:
        source_name = candidate_source_name or Path(candidate_source_path).stem
        candidate_library_dir = output_root / "candidate_library"
        candidate_artifacts = CandidateLibraryBuilder(output_dir=candidate_library_dir).build(
            candidate_source_path,
            dataset_id=dataset_id,
            source_name=source_name,
        )
        candidate_pool = candidate_artifacts.candidate_library_csv
        candidate_library_rows = candidate_artifacts.output_count

    verification_level = _verification_level(evidence_mode)
    run_metadata = {
        "evidence_mode": evidence_mode,
        "verification_level": verification_level,
        "publication_grade": evidence_mode == "external-cached",
        "source_columns_is_smoke_only": evidence_mode == "source-columns",
        "input_path": str(Path(input_path)),
        "max_rows": max_rows,
        "candidate_pool_contract_version": CANDIDATE_LIBRARY_CONTRACT_VERSION,
        "research_package_runner": "run_research_package",
    }
    if evidence_mode == "external-cached":
        run_metadata["cache_dir"] = str(cache_root)
    if candidate_pool is not None:
        run_metadata["candidate_pool_path"] = str(candidate_pool)

    discovery_artifacts = VerifiedDiscoveryWorkflow(
        output_dir=verified_discovery_dir,
        authenticator=authenticator,
    ).run_from_dataframe(
        df,
        dataset_id=dataset_id,
        candidate_pool=candidate_pool,
        top_k=top_k,
        min_verified_rows=min_verified_rows,
        run_metadata=run_metadata,
    )

    report_inputs = _report_inputs(
        verified_discovery_dir,
        top_n=verified_discovery_top_n or top_k,
        evidence_context={
            "evidence_mode": evidence_mode,
            "verification_level": verification_level,
            "publication_grade": evidence_mode == "external-cached",
            "source_columns_is_smoke_only": evidence_mode == "source-columns",
            "max_rows": max_rows,
            "max_rows_is_smoke_only": max_rows is not None,
            "metric_scope": "training_only",
            "candidate_pool_contract_version": CANDIDATE_LIBRARY_CONTRACT_VERSION,
        },
    )
    main_bundle = TopJournalReport(
        report_inputs["results"],
        report_inputs["artifacts"],
        output_dir=report_dir / "main_text",
        quality_target=report_quality_target,
    ).generate()
    si_path = SIGenerator(
        report_inputs["results"],
        report_inputs["artifacts"],
        output_dir=report_dir / "si",
    ).generate()

    root_manifest_path = output_root / "provenance_manifest.json"
    generate_root_provenance_manifest(
        verified_discovery_dir,
        main_bundle.path.parent,
        si_path.parent,
        candidate_library_dir=candidate_library_dir,
        output_path=root_manifest_path,
    )

    package_manifest_path = output_root / "package_manifest.json"
    _write_json(
        _package_manifest(
            dataset_id=dataset_id,
            output_root=output_root,
            evidence_mode=evidence_mode,
            discovery_artifacts=discovery_artifacts,
            candidate_library_dir=candidate_library_dir,
            candidate_library_rows=candidate_library_rows,
            report_dir=report_dir,
            report_quality_score=main_bundle.quality_score,
            root_manifest_path=root_manifest_path,
            package_manifest_path=package_manifest_path,
            max_rows=max_rows,
            candidate_pool_contract_version=CANDIDATE_LIBRARY_CONTRACT_VERSION,
        ),
        package_manifest_path,
    )
    return ResearchPackageArtifacts(
        output_dir=output_root,
        verified_discovery_dir=verified_discovery_dir,
        candidate_library_dir=candidate_library_dir,
        report_dir=report_dir,
        root_provenance_manifest_json=root_manifest_path,
        package_manifest_json=package_manifest_path,
    )


def _report_inputs(
    verified_discovery_dir: Path,
    *,
    top_n: int,
    evidence_context: dict[str, Any],
) -> dict[str, Any]:
    metrics = _read_json(verified_discovery_dir / "model" / "model_metrics.json")
    workflow = _read_json(verified_discovery_dir / "workflow_manifest.json")
    feature_columns = metrics.get("feature_columns", [])
    result = {
        "agent_id": "verified_discovery_model",
        "status": "success",
        "n_samples": metrics.get("train_rows", workflow.get("verified_rows", 0)),
        "n_features": len(feature_columns),
        "duration_sec": 0.0,
        "config": {
            "layer1": {"method_id": "strict_verified_data"},
            "layer2": {"method_id": "default_smiles_features"},
            "layer3": {"method_id": metrics.get("model_class", "model")},
            "layer4": {"method_id": "training_only_fit"},
            "layer5": {"method_id": "candidate_discovery_report"},
            "target": metrics.get("target_column", "delta_pce"),
            "baseline_as_feature": False,
            "_hash": f"verified_discovery_{workflow.get('dataset_id', 'dataset')}",
        },
        "metrics": {
            "r2": metrics.get("train_r2", 0.0),
            "rmse": metrics.get("train_rmse"),
            "mae": metrics.get("train_mae"),
            "pearson_r": metrics.get("train_pearson_r"),
            "metric_scope": "training_only",
        },
    }
    artifacts = {
        "verified_discovery_artifact_dir": verified_discovery_dir,
        "verified_discovery_top_n": top_n,
        "evidence_context": evidence_context,
        "multi_model_results": [
            {
                "name": "Verified discovery model",
                "agent_id": result["agent_id"],
                "r2": result["metrics"]["r2"],
                "rmse": result["metrics"]["rmse"],
                "mae": result["metrics"]["mae"],
                "pearson_r": result["metrics"]["pearson_r"],
                "n_samples": result["n_samples"],
                "n_features": result["n_features"],
            }
        ],
    }
    return {"results": [result], "artifacts": artifacts}


def _package_manifest(
    *,
    dataset_id: str,
    output_root: Path,
    evidence_mode: str,
    discovery_artifacts,
    candidate_library_dir: Path | None,
    candidate_library_rows: int | None,
    report_dir: Path,
    report_quality_score: float,
    root_manifest_path: Path,
    package_manifest_path: Path,
    max_rows: int | None,
    candidate_pool_contract_version: str,
) -> dict[str, Any]:
    verification_level = _verification_level(evidence_mode)
    return {
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "package_runner": "run_research_package",
        "evidence_mode": evidence_mode,
        "verification_level": verification_level,
        "publication_grade": evidence_mode == "external-cached",
        "source_columns_is_smoke_only": evidence_mode == "source-columns",
        "max_rows": max_rows,
        "max_rows_is_smoke_only": max_rows is not None,
        "candidate_pool_contract_version": candidate_pool_contract_version,
        "verified_rows": discovery_artifacts.verified_rows,
        "quarantine_rows": discovery_artifacts.quarantine_rows,
        "ranked_candidates": discovery_artifacts.ranked_candidates,
        "candidate_library_rows": candidate_library_rows,
        "report_quality_score": report_quality_score,
        "outputs": {
            "verified_discovery_dir": _relative(discovery_artifacts.output_dir, output_root),
            "candidate_library_dir": _relative(candidate_library_dir, output_root) if candidate_library_dir else None,
            "report_dir": _relative(report_dir, output_root),
            "main_text_report_md": _relative(report_dir / "main_text" / "main_text_report.md", output_root),
            "supporting_information_md": _relative(report_dir / "si" / "supporting_information.md", output_root),
            "root_provenance_manifest_json": _relative(root_manifest_path, output_root),
            "package_manifest_json": _relative(package_manifest_path, output_root),
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _verification_level(evidence_mode: str) -> str:
    return "external_cached" if evidence_mode == "external-cached" else "source_columns_only"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/XLSX PSC source table.")
    parser.add_argument("--output-dir", required=True, help="Directory for the complete research package.")
    parser.add_argument("--dataset-id", required=True, help="Stable package dataset/run identifier.")
    parser.add_argument("--evidence-mode", choices=("external-cached", "source-columns"), default="external-cached")
    parser.add_argument("--candidate-source", help="Optional CSV/XLSX source table for external candidates.")
    parser.add_argument("--candidate-source-name", help="Source name for candidate-source normalization.")
    parser.add_argument("--cache-dir", help="External evidence cache directory.")
    parser.add_argument("--max-rows", type=int, help="Optional row cap for smoke runs.")
    parser.add_argument("--min-verified-rows", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--report-quality-target", choices=("standard", "top-journal"), default="top-journal")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = run_research_package(
        input_path=args.input,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        evidence_mode=args.evidence_mode,
        candidate_source_path=args.candidate_source,
        candidate_source_name=args.candidate_source_name,
        cache_dir=args.cache_dir,
        max_rows=args.max_rows,
        min_verified_rows=args.min_verified_rows,
        top_k=args.top_k,
        report_quality_target=args.report_quality_target,
    )
    print(f"[research-package] package_manifest={artifacts.package_manifest_json}")
    print(f"[research-package] root_provenance_manifest={artifacts.root_provenance_manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
