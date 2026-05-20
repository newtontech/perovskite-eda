"""Run the verified PSC candidate-discovery research package.

This CLI connects the verified dataset/discovery workflow, optional external
candidate-library normalization, report/SI generation, and root provenance
manifest into one reproducible artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from closed_loop.feedback_ingestion import ExperimentalFeedbackIngestionArtifacts, ingest_experimental_feedback
from closed_loop.iteration_packet import generate_iteration_packet
from data.source_completeness import (
    summarize_source_completeness,
    write_source_completeness_artifacts,
)
from reporting.root_provenance_manifest import generate_root_provenance_manifest
from reporting.si_generator import SIGenerator
from reporting.top_journal_report import TopJournalReport
from run_verified_discovery import build_authenticator, default_cache_dir, load_input, publication_grade_gate
from screening.candidate_library_builder import CandidateLibraryBuilder
from screening.verified_candidate_discovery import CANDIDATE_LIBRARY_CONTRACT_VERSION
from screening.verified_discovery_workflow import VerifiedDiscoveryWorkflow


PACKAGE_MANIFEST_SCHEMA_VERSION = "research-package-manifest-v1"
EVIDENCE_CACHE_MANIFEST_SCHEMA_VERSION = "evidence-cache-manifest-v1"
EVIDENCE_CACHE_FILENAMES = {
    "reference_cache": "reference_cache.json",
    "molecule_cache": "molecule_cache.json",
}
INPUT_HASH_POLICY = {
    "algorithm": "sha256",
    "digest": "full_hex",
}


@dataclass(frozen=True)
class ResearchPackageArtifacts:
    """Top-level paths emitted by a research package run."""

    output_dir: Path
    verified_discovery_dir: Path
    source_completeness_dir: Path
    evidence_cache_dir: Path | None
    candidate_library_dir: Path | None
    closed_loop_iteration_dir: Path
    experimental_feedback_dir: Path | None
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
    input_scope: str = "selected-subset",
    min_verified_rows: int = 10,
    top_k: int = 100,
    report_quality_target: str = "top-journal",
    verified_discovery_top_n: int | None = None,
    iteration_index: int = 0,
    iteration_top_k: int = 10,
    experimental_feedback_csv: str | Path | None = None,
    experimental_feedback_data_scope: str = "external_lab",
) -> ResearchPackageArtifacts:
    """Generate the full verified-discovery research artifact package."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    verified_discovery_dir = output_root / "verified_discovery"
    report_dir = output_root / "report_bundle"

    df = load_input(input_path, max_rows=max_rows)
    source_completeness_artifacts = write_source_completeness_artifacts(
        summarize_source_completeness(
            df,
            dataset_id=dataset_id,
            source_name=Path(input_path).stem,
            max_rows=max_rows,
        ),
        output_root / "source_completeness",
    )
    cache_root = Path(cache_dir) if cache_dir else default_cache_dir(dataset_id)
    evidence_cache_manifest_path: Path | None = None
    evidence_cache_dir: Path | None = None
    if evidence_mode == "external-cached":
        evidence_cache_dir = output_root / "evidence_cache"
        evidence_cache_manifest_path = evidence_cache_dir / "evidence_cache_manifest.json"
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
    dataset_publication_grade, dataset_publication_grade_reason = publication_grade_gate(
        evidence_mode=evidence_mode,
        input_scope=input_scope,
        max_rows=max_rows,
    )
    candidate_library_publication_grade = candidate_source_path is None
    candidate_library_publication_grade_reason = (
        "no offline candidate source" if candidate_library_publication_grade else "candidate library is offline-normalized"
    )
    publication_grade = dataset_publication_grade and candidate_library_publication_grade
    publication_grade_reasons = _publication_grade_reasons(
        dataset_publication_grade=dataset_publication_grade,
        dataset_publication_grade_reason=dataset_publication_grade_reason,
        candidate_library_publication_grade=candidate_library_publication_grade,
        candidate_library_publication_grade_reason=candidate_library_publication_grade_reason,
    )
    publication_grade_reason = _publication_grade_reason(
        publication_grade=publication_grade,
        publication_grade_reasons=publication_grade_reasons,
    )
    run_metadata = {
        "evidence_mode": evidence_mode,
        "verification_level": verification_level,
        "publication_grade": publication_grade,
        "publication_grade_reason": publication_grade_reason,
        "publication_grade_reasons": publication_grade_reasons,
        "dataset_publication_grade": dataset_publication_grade,
        "dataset_publication_grade_reason": dataset_publication_grade_reason,
        "candidate_library_publication_grade": candidate_library_publication_grade,
        "candidate_library_publication_grade_reason": candidate_library_publication_grade_reason,
        "input_scope": input_scope,
        "source_columns_is_smoke_only": evidence_mode == "source-columns",
        "input_path": str(Path(input_path)),
        "max_rows": max_rows,
        "max_rows_is_smoke_only": max_rows is not None,
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
    if evidence_cache_manifest_path is not None:
        _write_json(
            _evidence_cache_manifest(dataset_id=dataset_id, cache_root=cache_root),
            evidence_cache_manifest_path,
        )
    closed_loop_artifacts = generate_iteration_packet(
        ranked_candidates_csv=discovery_artifacts.ranked_candidates_csv,
        output_dir=output_root / "closed_loop" / f"iteration_{iteration_index:03d}",
        dataset_id=dataset_id,
        iteration_index=iteration_index,
        top_k=min(iteration_top_k, max(discovery_artifacts.ranked_candidates, 1)),
    )
    experimental_feedback_artifacts: ExperimentalFeedbackIngestionArtifacts | None = None
    if experimental_feedback_csv is not None:
        experimental_feedback_artifacts = ingest_experimental_feedback(
            iteration_dir=closed_loop_artifacts.output_dir,
            feedback_csv=experimental_feedback_csv,
            feedback_data_scope=experimental_feedback_data_scope,
        )

    report_inputs = _report_inputs(
        verified_discovery_dir,
        top_n=verified_discovery_top_n or top_k,
        source_completeness=source_completeness_artifacts.summary,
        evidence_context={
            "evidence_mode": evidence_mode,
            "verification_level": verification_level,
            "publication_grade": publication_grade,
            "publication_grade_reason": publication_grade_reason,
            "publication_grade_reasons": publication_grade_reasons,
            "dataset_publication_grade": dataset_publication_grade,
            "dataset_publication_grade_reason": dataset_publication_grade_reason,
            "candidate_library_publication_grade": candidate_library_publication_grade,
            "candidate_library_publication_grade_reason": candidate_library_publication_grade_reason,
            "input_scope": input_scope,
            "source_columns_is_smoke_only": evidence_mode == "source-columns",
            "max_rows": max_rows,
            "max_rows_is_smoke_only": max_rows is not None,
            "metric_scope": "training_and_internal_holdout",
            "internal_validation": True,
            "external_validation": False,
            "validation_scope": "internal_random_holdout_verified_rows",
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
    package_manifest_path = output_root / "package_manifest.json"
    closure_readiness_report_path = output_root / "closure_readiness_report.json"
    _write_json(
        _closure_readiness_report(
            dataset_id=dataset_id,
            evidence_mode=evidence_mode,
            input_scope=input_scope,
            max_rows=max_rows,
            publication_grade=publication_grade,
            publication_grade_reasons=publication_grade_reasons,
            dataset_publication_grade=dataset_publication_grade,
            candidate_library_publication_grade=candidate_library_publication_grade,
            candidate_library_rows=candidate_library_rows,
            evidence_cache_manifest_path=evidence_cache_manifest_path,
            closed_loop_iteration_dir=closed_loop_artifacts.output_dir,
            experimental_feedback_artifacts=experimental_feedback_artifacts,
            validation_metrics_path=discovery_artifacts.model_validation_metrics_json,
            scaffold_validation_metrics_path=discovery_artifacts.model_scaffold_validation_metrics_json,
        ),
        closure_readiness_report_path,
    )
    _write_json(
        _package_manifest(
            dataset_id=dataset_id,
            output_root=output_root,
            evidence_mode=evidence_mode,
            dataset_publication_grade=dataset_publication_grade,
            dataset_publication_grade_reason=dataset_publication_grade_reason,
            candidate_library_publication_grade=candidate_library_publication_grade,
            candidate_library_publication_grade_reason=candidate_library_publication_grade_reason,
            publication_grade=publication_grade,
            publication_grade_reason=publication_grade_reason,
            publication_grade_reasons=publication_grade_reasons,
            input_scope=input_scope,
            input_path=Path(input_path),
            candidate_source_path=Path(candidate_source_path) if candidate_source_path is not None else None,
            candidate_source_name=candidate_source_name,
            experimental_feedback_csv=Path(experimental_feedback_csv) if experimental_feedback_csv is not None else None,
            experimental_feedback_data_scope=experimental_feedback_data_scope,
            cache_dir=cache_root if evidence_mode == "external-cached" else None,
            min_verified_rows=min_verified_rows,
            top_k=top_k,
            report_quality_target=report_quality_target,
            verified_discovery_top_n=verified_discovery_top_n,
            iteration_index=iteration_index,
            iteration_top_k=iteration_top_k,
            discovery_artifacts=discovery_artifacts,
            source_completeness_dir=source_completeness_artifacts.output_dir,
            evidence_cache_manifest_path=evidence_cache_manifest_path,
            candidate_library_dir=candidate_library_dir,
            candidate_library_rows=candidate_library_rows,
            closed_loop_iteration_dir=closed_loop_artifacts.output_dir,
            experimental_feedback_artifacts=experimental_feedback_artifacts,
            closure_readiness_report_path=closure_readiness_report_path,
            report_dir=report_dir,
            report_quality_score=main_bundle.quality_score,
            root_manifest_path=root_manifest_path,
            package_manifest_path=package_manifest_path,
            max_rows=max_rows,
            candidate_pool_contract_version=CANDIDATE_LIBRARY_CONTRACT_VERSION,
        ),
        package_manifest_path,
    )
    generate_root_provenance_manifest(
        verified_discovery_dir,
        main_bundle.path.parent,
        si_path.parent,
        candidate_library_dir=candidate_library_dir,
        source_completeness_dir=source_completeness_artifacts.output_dir,
        closed_loop_iteration_dir=closed_loop_artifacts.output_dir,
        experimental_feedback_dir=experimental_feedback_artifacts.output_dir
        if experimental_feedback_artifacts is not None
        else None,
        closure_readiness_report_path=closure_readiness_report_path,
        evidence_cache_manifest_path=evidence_cache_manifest_path,
        package_manifest_path=package_manifest_path,
        input_path=input_path,
        candidate_source_path=candidate_source_path,
        output_path=root_manifest_path,
    )
    return ResearchPackageArtifacts(
        output_dir=output_root,
        verified_discovery_dir=verified_discovery_dir,
        source_completeness_dir=source_completeness_artifacts.output_dir,
        evidence_cache_dir=evidence_cache_dir,
        candidate_library_dir=candidate_library_dir,
        closed_loop_iteration_dir=closed_loop_artifacts.output_dir,
        experimental_feedback_dir=experimental_feedback_artifacts.output_dir
        if experimental_feedback_artifacts is not None
        else None,
        report_dir=report_dir,
        root_provenance_manifest_json=root_manifest_path,
        package_manifest_json=package_manifest_path,
    )


def _report_inputs(
    verified_discovery_dir: Path,
    *,
    top_n: int,
    source_completeness: dict[str, Any],
    evidence_context: dict[str, Any],
) -> dict[str, Any]:
    metrics = _read_json(verified_discovery_dir / "model" / "model_metrics.json")
    validation_metrics = _read_json(verified_discovery_dir / "model" / "model_validation_metrics.json")
    scaffold_validation_metrics = _read_json(
        verified_discovery_dir / "model" / "model_scaffold_validation_metrics.json"
    )
    workflow = _read_json(verified_discovery_dir / "workflow_manifest.json")
    feature_columns = metrics.get("feature_columns", [])
    evidence_context = {
        **evidence_context,
        "chemical_generalization_validation": scaffold_validation_metrics.get("status") == "passed",
        "chemical_generalization_validation_scope": scaffold_validation_metrics.get("validation_scope"),
        "chemical_generalization_external_validation": scaffold_validation_metrics.get("external_validation", False),
    }
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
            "layer4": {"method_id": validation_metrics.get("split_strategy", "internal_random_holdout")},
            "layer5": {"method_id": "candidate_discovery_report"},
            "target": metrics.get("target_column", "delta_pce"),
            "baseline_as_feature": False,
            "_hash": f"verified_discovery_{workflow.get('dataset_id', 'dataset')}",
        },
        "metrics": {
            "r2": validation_metrics.get("validation_r2")
            if validation_metrics.get("validation_r2") is not None
            else metrics.get("train_r2", 0.0),
            "rmse": validation_metrics.get("validation_rmse"),
            "mae": validation_metrics.get("validation_mae"),
            "pearson_r": validation_metrics.get("validation_pearson_r"),
            "metric_scope": "training_and_internal_holdout",
            "display_r2_fallback": validation_metrics.get("validation_r2") is None,
        },
    }
    artifacts = {
        "verified_discovery_artifact_dir": verified_discovery_dir,
        "verified_discovery_top_n": top_n,
        "model_validation_metrics": validation_metrics,
        "model_scaffold_validation_metrics": scaffold_validation_metrics,
        "source_completeness": source_completeness,
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
    dataset_publication_grade: bool,
    dataset_publication_grade_reason: str,
    candidate_library_publication_grade: bool,
    candidate_library_publication_grade_reason: str,
    publication_grade: bool,
    publication_grade_reason: str,
    publication_grade_reasons: list[str],
    input_scope: str,
    input_path: Path,
    candidate_source_path: Path | None,
    candidate_source_name: str | None,
    experimental_feedback_csv: Path | None,
    experimental_feedback_data_scope: str,
    cache_dir: Path | None,
    min_verified_rows: int,
    top_k: int,
    report_quality_target: str,
    verified_discovery_top_n: int | None,
    iteration_index: int,
    iteration_top_k: int,
    discovery_artifacts,
    source_completeness_dir: Path,
    evidence_cache_manifest_path: Path | None,
    candidate_library_dir: Path | None,
    candidate_library_rows: int | None,
    closed_loop_iteration_dir: Path,
    experimental_feedback_artifacts: ExperimentalFeedbackIngestionArtifacts | None,
    closure_readiness_report_path: Path,
    report_dir: Path,
    report_quality_score: float,
    root_manifest_path: Path,
    package_manifest_path: Path,
    max_rows: int | None,
    candidate_pool_contract_version: str,
) -> dict[str, Any]:
    verification_level = _verification_level(evidence_mode)
    runner_args = {
        "input": str(input_path),
        "output_dir": str(output_root),
        "dataset_id": dataset_id,
        "evidence_mode": evidence_mode,
        "candidate_source": str(candidate_source_path) if candidate_source_path is not None else None,
        "candidate_source_name": candidate_source_name,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "max_rows": max_rows,
        "input_scope": input_scope,
        "min_verified_rows": min_verified_rows,
        "top_k": top_k,
        "report_quality_target": report_quality_target,
        "verified_discovery_top_n": verified_discovery_top_n,
        "iteration_index": iteration_index,
        "iteration_top_k": iteration_top_k,
        "experimental_feedback_csv": str(experimental_feedback_csv) if experimental_feedback_csv is not None else None,
        "experimental_feedback_data_scope": experimental_feedback_data_scope,
    }
    source_table_facts = _file_facts(input_path)
    candidate_source_facts = _file_facts(candidate_source_path) if candidate_source_path else None
    if candidate_source_facts is not None:
        candidate_source_facts["source_name"] = candidate_source_name
    experimental_feedback_facts = _file_facts(experimental_feedback_csv) if experimental_feedback_csv else None
    if experimental_feedback_facts is not None:
        experimental_feedback_facts["feedback_data_scope"] = experimental_feedback_data_scope
    outputs = {
        "verified_discovery_dir": _relative(discovery_artifacts.output_dir, output_root),
        "source_completeness_dir": _relative(source_completeness_dir, output_root),
        "source_completeness_json": _relative(source_completeness_dir / "source_completeness.json", output_root),
        "source_completeness_csv": _relative(source_completeness_dir / "source_completeness.csv", output_root),
        "source_completeness_md": _relative(source_completeness_dir / "source_completeness.md", output_root),
        "candidate_library_dir": _relative(candidate_library_dir, output_root) if candidate_library_dir else None,
        "closed_loop_iteration_dir": _relative(closed_loop_iteration_dir, output_root),
        "closed_loop_iteration_manifest_json": _relative(
            closed_loop_iteration_dir / "iteration_manifest.json",
            output_root,
        ),
        "closed_loop_selected_candidates_csv": _relative(
            closed_loop_iteration_dir / "selected_candidates.csv",
            output_root,
        ),
        "closed_loop_candidate_packet_md": _relative(
            closed_loop_iteration_dir / "candidate_packet.md",
            output_root,
        ),
        "closed_loop_feedback_schema_json": _relative(
            closed_loop_iteration_dir / "feedback_schema.json",
            output_root,
        ),
        "closed_loop_feedback_template_csv": _relative(
            closed_loop_iteration_dir / "feedback_template.csv",
            output_root,
        ),
        "closed_loop_verification_report_json": _relative(
            closed_loop_iteration_dir / "verification_report.json",
            output_root,
        ),
        "closure_readiness_report_json": _relative(closure_readiness_report_path, output_root),
        "report_dir": _relative(report_dir, output_root),
        "main_text_report_md": _relative(report_dir / "main_text" / "main_text_report.md", output_root),
        "supporting_information_md": _relative(report_dir / "si" / "supporting_information.md", output_root),
        "root_provenance_manifest_json": _relative(root_manifest_path, output_root),
        "package_manifest_json": _relative(package_manifest_path, output_root),
    }
    if evidence_cache_manifest_path is not None:
        outputs["evidence_cache_dir"] = _relative(evidence_cache_manifest_path.parent, output_root)
        outputs["evidence_cache_manifest_json"] = _relative(evidence_cache_manifest_path, output_root)
    if candidate_library_dir is not None:
        outputs["candidate_external_evidence_audit_json"] = _relative(
            candidate_library_dir / "external_evidence_audit.json",
            output_root,
        )
        outputs["candidate_external_evidence_audit_md"] = _relative(
            candidate_library_dir / "external_evidence_audit.md",
            output_root,
        )
    if experimental_feedback_artifacts is not None:
        outputs["experimental_feedback_dir"] = _relative(experimental_feedback_artifacts.output_dir, output_root)
        outputs["experimental_feedback_validated_csv"] = _relative(
            experimental_feedback_artifacts.validated_feedback_csv,
            output_root,
        )
        outputs["experimental_feedback_vs_prediction_csv"] = _relative(
            experimental_feedback_artifacts.feedback_vs_prediction_csv,
            output_root,
        )
        outputs["experimental_feedback_ingestion_manifest_json"] = _relative(
            experimental_feedback_artifacts.feedback_ingestion_manifest_json,
            output_root,
        )
        outputs["experimental_feedback_validation_report_json"] = _relative(
            experimental_feedback_artifacts.feedback_validation_report_json,
            output_root,
        )
    outputs["model_validation_metrics_json"] = _relative(
        discovery_artifacts.model_validation_metrics_json,
        output_root,
    )
    outputs["model_validation_predictions_csv"] = _relative(
        discovery_artifacts.model_validation_predictions_csv,
        output_root,
    )
    outputs["model_scaffold_validation_metrics_json"] = _relative(
        discovery_artifacts.model_scaffold_validation_metrics_json,
        output_root,
    )
    outputs["model_scaffold_validation_predictions_csv"] = _relative(
        discovery_artifacts.model_scaffold_validation_predictions_csv,
        output_root,
    )
    return {
        "schema_version": PACKAGE_MANIFEST_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "package_runner": "run_research_package",
        "runner": {
            "name": "run_research_package",
            "args": runner_args,
        },
        "runner_args": runner_args,
        "input_hash_policy": INPUT_HASH_POLICY,
        "inputs": {
            "source_table": source_table_facts,
            "candidate_source": candidate_source_facts,
            "experimental_feedback": experimental_feedback_facts,
        },
        "evidence_mode": evidence_mode,
        "verification_level": verification_level,
        "publication_grade": publication_grade,
        "publication_grade_reason": publication_grade_reason,
        "publication_grade_reasons": publication_grade_reasons,
        "dataset_publication_grade": dataset_publication_grade,
        "dataset_publication_grade_reason": dataset_publication_grade_reason,
        "candidate_library_publication_grade": candidate_library_publication_grade,
        "candidate_library_publication_grade_reason": candidate_library_publication_grade_reason,
        "input_scope": input_scope,
        "source_columns_is_smoke_only": evidence_mode == "source-columns",
        "max_rows": max_rows,
        "max_rows_is_smoke_only": max_rows is not None,
        "candidate_pool_contract_version": candidate_pool_contract_version,
        "verified_rows": discovery_artifacts.verified_rows,
        "quarantine_rows": discovery_artifacts.quarantine_rows,
        "ranked_candidates": discovery_artifacts.ranked_candidates,
        "candidate_library_rows": candidate_library_rows,
        "report_quality_score": report_quality_score,
        "outputs": outputs,
    }


def _closure_readiness_report(
    *,
    dataset_id: str,
    evidence_mode: str,
    input_scope: str,
    max_rows: int | None,
    publication_grade: bool,
    publication_grade_reasons: list[str],
    dataset_publication_grade: bool,
    candidate_library_publication_grade: bool,
    candidate_library_rows: int | None,
    evidence_cache_manifest_path: Path | None,
    closed_loop_iteration_dir: Path,
    experimental_feedback_artifacts: ExperimentalFeedbackIngestionArtifacts | None,
    validation_metrics_path: Path,
    scaffold_validation_metrics_path: Path,
) -> dict[str, Any]:
    closure_blockers: list[str] = []
    chemical_generalization_gate = _chemical_generalization_gate(scaffold_validation_metrics_path)
    experimental_feedback_gate = _experimental_feedback_gate(experimental_feedback_artifacts)
    gates = {
        "publication_grade": {
            "passed": publication_grade,
            "reason": "publication_grade is true" if publication_grade else "publication_grade is false",
        },
        "dataset_publication_grade": {
            "passed": dataset_publication_grade,
            "reason": "dataset gate passed" if dataset_publication_grade else "dataset gate failed",
        },
        "candidate_library_publication_grade": {
            "passed": candidate_library_publication_grade,
            "reason": (
                "candidate library is publication-grade"
                if candidate_library_publication_grade
                else "candidate library is offline-normalized"
            ),
        },
        "full_source_scope": {
            "passed": input_scope == "full-source" and max_rows is None,
            "reason": f"input_scope={input_scope}, max_rows={max_rows}",
        },
        "evidence_cache": {
            "passed": evidence_mode == "external-cached" and evidence_cache_manifest_path is not None,
            "reason": "external-cached evidence manifest present"
            if evidence_mode == "external-cached" and evidence_cache_manifest_path is not None
            else "external-cached evidence manifest absent",
        },
        "validation_artifacts": {
            "passed": validation_metrics_path.is_file(),
            "reason": "internal holdout validation artifact present",
        },
        "chemical_generalization_validation": chemical_generalization_gate,
        "external_validation": {
            "passed": False,
            "reason": "no independent external, temporal, or experimental validation artifact",
        },
        "closed_loop_packet": {
            "passed": (closed_loop_iteration_dir / "verification_report.json").is_file(),
            "reason": "closed-loop handoff packet integrity artifact present",
        },
        "experimental_feedback": {
            "passed": experimental_feedback_gate["passed"],
            "reason": experimental_feedback_gate["reason"],
            **{
                key: value
                for key, value in experimental_feedback_gate.items()
                if key not in {"passed", "reason"}
            },
        },
    }
    for gate_name, gate in gates.items():
        if not gate["passed"]:
            closure_blockers.append(f"{gate_name}: {gate['reason']}")
    false_reasons = list(publication_grade_reasons)
    if not publication_grade and not false_reasons:
        false_reasons = ["publication_grade is false"]
    return {
        "schema_version": "closure-readiness-report-v1",
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "overall_ready": not closure_blockers,
        "publication_grade": publication_grade,
        "publication_grade_false_reasons": false_reasons,
        "blocking_reasons": closure_blockers,
        "gates": gates,
    }


def _experimental_feedback_gate(artifacts: ExperimentalFeedbackIngestionArtifacts | None) -> dict[str, Any]:
    if artifacts is None:
        return {
            "passed": False,
            "reason": "no real experimental feedback measurements have been ingested",
        }
    report_path = artifacts.feedback_validation_report_json
    try:
        report = _read_json(report_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "passed": False,
            "reason": f"experimental feedback validation report unreadable: {exc}",
        }
    feedback_data_scope = report.get("feedback_data_scope")
    if report.get("status") != "passed":
        return {
            "passed": False,
            "reason": "experimental feedback validation report did not pass",
            "feedback_data_scope": feedback_data_scope,
        }
    if feedback_data_scope != "external_lab":
        return {
            "passed": False,
            "reason": f"feedback_data_scope={feedback_data_scope!r} is not external_lab",
            "feedback_data_scope": feedback_data_scope,
        }
    return {
        "passed": True,
        "reason": "validated external_lab feedback artifact present",
        "feedback_data_scope": feedback_data_scope,
        "validated_feedback_rows": report.get("validated_feedback_rows"),
    }


def _chemical_generalization_gate(scaffold_validation_metrics_path: Path) -> dict[str, Any]:
    if not scaffold_validation_metrics_path.is_file():
        return {
            "passed": False,
            "reason": "chemical-group holdout validation artifact absent",
        }
    try:
        metrics = _read_json(scaffold_validation_metrics_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "passed": False,
            "reason": f"chemical-group holdout validation artifact unreadable: {exc}",
        }
    train_groups = [str(group) for group in metrics.get("train_groups", [])]
    validation_groups = [str(group) for group in metrics.get("validation_groups", [])]
    overlap = sorted(set(train_groups) & set(validation_groups))
    group_method = metrics.get("group_method") if isinstance(metrics.get("group_method"), dict) else {}
    validation_rows = int(metrics.get("validation_rows") or 0)
    passed = (
        metrics.get("status") == "passed"
        and metrics.get("chemical_generalization_diagnostic") is True
        and metrics.get("external_validation") is False
        and metrics.get("independent_external_validation") is False
        and metrics.get("random_holdout") is False
        and metrics.get("chemical_generalization_claim") == "not_established"
        and group_method.get("rdkit_available") is True
        and group_method.get("uses_smiles_skeleton_fallback") is False
        and validation_rows >= 2
        and metrics.get("validation_r2") is not None
        and bool(train_groups)
        and bool(validation_groups)
        and not overlap
    )
    if passed:
        return {
            "passed": True,
            "reason": "RDKit chemical-group holdout diagnostic artifact present with disjoint groups",
        }
    if metrics.get("status") == "skipped":
        return {
            "passed": False,
            "reason": f"chemical-group holdout skipped: {metrics.get('skip_reason', 'reason unavailable')}",
        }
    return {
        "passed": False,
        "reason": (
            "chemical-group holdout diagnostic failed "
            f"(status={metrics.get('status')!r}, validation_rows={validation_rows}, "
            f"validation_r2={metrics.get('validation_r2')!r}, "
            f"rdkit_available={group_method.get('rdkit_available')!r}, "
            f"uses_smiles_skeleton_fallback={group_method.get('uses_smiles_skeleton_fallback')!r}, "
            f"overlap_groups={overlap})"
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_facts(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "sha256": _sha256(path) if exists and path.is_file() else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_cache_manifest(*, dataset_id: str, cache_root: Path) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_CACHE_MANIFEST_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "artifact_policy": "external-untracked-hashed-artifact",
        "cache_dir": str(cache_root),
        "hash_policy": INPUT_HASH_POLICY,
        "cache_files": {
            cache_key: _cache_file_record(cache_root / filename, filename=filename)
            for cache_key, filename in EVIDENCE_CACHE_FILENAMES.items()
        },
    }


def _cache_file_record(path: Path, *, filename: str) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "filename": filename,
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": _sha256(path) if exists else None,
        "entry_count": _json_object_entry_count(path) if exists else None,
    }


def _json_object_entry_count(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return len(payload) if isinstance(payload, dict) else None


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


def _publication_grade_reason(
    *,
    publication_grade: bool,
    publication_grade_reasons: list[str],
) -> str:
    if publication_grade:
        return "all publication-grade gates passed"
    if publication_grade_reasons:
        return publication_grade_reasons[0]
    return "publication-grade gates failed"


def _publication_grade_reasons(
    *,
    dataset_publication_grade: bool,
    dataset_publication_grade_reason: str,
    candidate_library_publication_grade: bool,
    candidate_library_publication_grade_reason: str,
) -> list[str]:
    reasons: list[str] = []
    if not dataset_publication_grade:
        reasons.append(dataset_publication_grade_reason)
    if not candidate_library_publication_grade:
        reasons.append(candidate_library_publication_grade_reason)
    return reasons


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
    parser.add_argument(
        "--input-scope",
        choices=("selected-subset", "full-source"),
        default="selected-subset",
        help="Declare whether the input is the full source table or a selected subset.",
    )
    parser.add_argument("--min-verified-rows", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--iteration-index", type=int, default=0)
    parser.add_argument("--iteration-top-k", type=int, default=10)
    parser.add_argument("--experimental-feedback-csv", help="Optional user-supplied experimental feedback CSV.")
    parser.add_argument(
        "--experimental-feedback-data-scope",
        default="external_lab",
        help="Scope label for experimental feedback provenance.",
    )
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
        input_scope=args.input_scope,
        min_verified_rows=args.min_verified_rows,
        top_k=args.top_k,
        report_quality_target=args.report_quality_target,
        iteration_index=args.iteration_index,
        iteration_top_k=args.iteration_top_k,
        experimental_feedback_csv=args.experimental_feedback_csv,
        experimental_feedback_data_scope=args.experimental_feedback_data_scope,
    )
    print(f"[research-package] package_manifest={artifacts.package_manifest_json}")
    print(f"[research-package] root_provenance_manifest={artifacts.root_provenance_manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
