"""Verify that a generated research package satisfies the artifact contract.

The verifier is intentionally read-only. It checks that the package manifest,
root provenance manifest, and the expected scientific outputs agree with the
files on disk before a package is treated as a complete research deliverable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from closed_loop.iteration_packet import ClosedLoopPacketError, verify_iteration_packet


PACKAGE_MANIFEST_NAME = "package_manifest.json"
ROOT_PROVENANCE_NAME = "provenance_manifest.json"
PACKAGE_SCHEMA_VERSION = "research-package-manifest-v1"
ROOT_PROVENANCE_SCHEMA_VERSION = "root-provenance-manifest-v1"
EVIDENCE_CACHE_MANIFEST_SCHEMA_VERSION = "evidence-cache-manifest-v1"
EVIDENCE_CACHE_MANIFEST_PATH = "evidence_cache/evidence_cache_manifest.json"
CLOSURE_READINESS_REPORT_PATH = "closure_readiness_report.json"

REQUIRED_PACKAGE_OUTPUTS = {
    "verified_discovery_dir": "dir",
    "source_completeness_dir": "dir",
    "source_completeness_json": "file",
    "source_completeness_csv": "file",
    "source_completeness_md": "file",
    "report_dir": "dir",
    "main_text_report_md": "file",
    "supporting_information_md": "file",
    "closed_loop_iteration_dir": "dir",
    "closed_loop_iteration_manifest_json": "file",
    "closed_loop_selected_candidates_csv": "file",
    "closed_loop_candidate_packet_md": "file",
    "closed_loop_feedback_schema_json": "file",
    "closed_loop_feedback_template_csv": "file",
    "closed_loop_verification_report_json": "file",
    "closure_readiness_report_json": "file",
    "model_validation_metrics_json": "file",
    "model_validation_predictions_csv": "file",
    "model_scaffold_validation_metrics_json": "file",
    "model_scaffold_validation_predictions_csv": "file",
    "root_provenance_manifest_json": "file",
    "package_manifest_json": "file",
}

REQUIRED_ARTIFACT_FILES = (
    "package_manifest.json",
    "provenance_manifest.json",
    "verified_discovery/dataset/verified_train.csv",
    "verified_discovery/dataset/quarantine.csv",
    "verified_discovery/dataset/candidate_pool.csv",
    "verified_discovery/dataset/doi_manifest.json",
    "verified_discovery/dataset/provenance.json",
    "verified_discovery/dataset/data_audit_report.md",
    "verified_discovery/model/model_metrics.json",
    "verified_discovery/model/model_manifest.json",
    "verified_discovery/model/model_validation_metrics.json",
    "verified_discovery/model/model_validation_predictions.csv",
    "verified_discovery/model/model_scaffold_validation_metrics.json",
    "verified_discovery/model/model_scaffold_validation_predictions.csv",
    "verified_discovery/discovery/ranked_candidates.csv",
    "verified_discovery/discovery/candidate_discovery_manifest.json",
    "verified_discovery/discovery/candidate_discovery_audit.md",
    "verified_discovery/workflow_manifest.json",
    "source_completeness/source_completeness.json",
    "source_completeness/source_completeness.csv",
    "source_completeness/source_completeness.md",
    "report_bundle/main_text/main_text_report.md",
    "report_bundle/main_text/claim_ledger.json",
    "report_bundle/main_text/review_report.json",
    "report_bundle/main_text/run_manifest.json",
    "report_bundle/si/supporting_information.md",
    "closed_loop/iteration_000/iteration_manifest.json",
    "closed_loop/iteration_000/selected_candidates.csv",
    "closed_loop/iteration_000/candidate_packet.md",
    "closed_loop/iteration_000/feedback_schema.json",
    "closed_loop/iteration_000/feedback_template.csv",
    "closed_loop/iteration_000/verification_report.json",
    "closure_readiness_report.json",
)

CANDIDATE_LIBRARY_FILES = (
    "candidate_library/candidate_library.csv",
    "candidate_library/source_summary.json",
    "candidate_library/provenance.json",
)

EVIDENCE_CACHE_FILES = (
    ("reference_cache", "reference_cache.json"),
    ("molecule_cache", "molecule_cache.json"),
)

REQUIRED_WORKFLOW_OUTPUTS = {
    "verified_train_csv",
    "quarantine_csv",
    "candidate_pool_csv",
    "doi_manifest_json",
    "provenance_json",
    "data_audit_report_md",
    "model_metrics_json",
    "model_manifest_json",
    "model_validation_metrics_json",
    "model_validation_predictions_csv",
    "model_scaffold_validation_metrics_json",
    "model_scaffold_validation_predictions_csv",
    "ranked_candidates_csv",
    "discovery_manifest_json",
    "discovery_audit_report_md",
    "workflow_manifest_json",
}

JSON_FIELD_REQUIREMENTS = {
    "package_manifest.json": ("schema_version", "dataset_id", "outputs"),
    "provenance_manifest.json": ("schema_version", "artifacts"),
    "verified_discovery/dataset/doi_manifest.json": ("dataset_id", "references"),
    "verified_discovery/dataset/provenance.json": ("dataset_id", "outputs"),
    "verified_discovery/model/model_metrics.json": ("dataset_id", "train_rows"),
    "verified_discovery/model/model_manifest.json": ("dataset_id", "model_class"),
    "verified_discovery/model/model_validation_metrics.json": (
        "schema_version",
        "dataset_id",
        "validation_scope",
        "external_validation",
    ),
    "verified_discovery/model/model_scaffold_validation_metrics.json": (
        "schema_version",
        "dataset_id",
        "validation_scope",
        "split_strategy",
        "external_validation",
        "chemical_generalization_diagnostic",
        "chemical_group_column",
    ),
    "verified_discovery/discovery/candidate_discovery_manifest.json": ("dataset_id", "outputs"),
    "verified_discovery/workflow_manifest.json": ("dataset_id", "outputs"),
    "source_completeness/source_completeness.json": ("schema_version", "dataset_id", "groups"),
    "evidence_cache/evidence_cache_manifest.json": ("schema_version", "dataset_id", "cache_files"),
    "report_bundle/main_text/review_report.json": ("review", "claim_audit"),
    "report_bundle/main_text/run_manifest.json": ("generated_at", "best_model"),
    "candidate_library/source_summary.json": ("dataset_id", "output_rows"),
    "candidate_library/provenance.json": ("dataset_id", "validation"),
    "closed_loop/iteration_000/iteration_manifest.json": (
        "schema_version",
        "dataset_id",
        "selection_policy",
        "experimental_measurement_policy",
    ),
    "closed_loop/iteration_000/feedback_schema.json": ("schema_version", "properties"),
    "closed_loop/iteration_000/verification_report.json": ("schema_version", "status", "checks"),
    "closure_readiness_report.json": (
        "schema_version",
        "dataset_id",
        "overall_ready",
        "publication_grade",
        "publication_grade_false_reasons",
        "gates",
    ),
}

REQUIRED_ROOT_IDS = {
    "dataset": {
        "verified_train_csv",
        "quarantine_csv",
        "candidate_pool_csv",
        "doi_manifest_json",
        "provenance_json",
        "data_audit_report_md",
    },
    "model": {
        "model_metrics_json",
        "model_manifest_json",
        "model_validation_metrics_json",
        "model_validation_predictions_csv",
        "model_scaffold_validation_metrics_json",
        "model_scaffold_validation_predictions_csv",
    },
    "discovery": {"ranked_candidates_csv", "discovery_manifest_json"},
    "audit": {"discovery_audit_report_md"},
    "report": {"main_text_report_md", "run_manifest_json"},
    "SI": {"supporting_information_md"},
    "claim": {"claim_ledger_json"},
    "review": {"review_report_json"},
    "source_completeness": {
        "source_completeness_json",
        "source_completeness_csv",
        "source_completeness_md",
    },
    "closed_loop": {
        "candidate_packet_md",
        "feedback_schema_json",
        "feedback_template_csv",
        "iteration_manifest_json",
        "selected_candidates_csv",
        "verification_report_json",
    },
    "closure_readiness": {"closure_readiness_report_json"},
    "package": {"package_manifest_json"},
}

CANDIDATE_LIBRARY_ROOT_IDS = {
    "discovery": {"candidate_library:candidate_library_csv"},
    "audit": {
        "candidate_library:source_summary_json",
        "candidate_library:provenance_json",
    },
}

EVIDENCE_CACHE_ROOT_IDS = {
    "evidence_cache": {"evidence_cache_manifest_json"},
}


class ResearchPackageContractError(RuntimeError):
    """Raised when a research package violates the artifact contract."""


def verify_research_package(
    package_dir: str | Path,
    *,
    require_candidate_library: bool = False,
    require_evidence_cache: bool = False,
    require_closure_readiness: bool = False,
) -> dict[str, Any]:
    """Verify that a research package contains every required deliverable.

    Args:
        package_dir: Directory produced by ``run_research_package``.
        require_candidate_library: Require candidate-library artifacts even if
            the package manifest does not declare a candidate source.
        require_evidence_cache: Require external evidence cache JSON files.

    Returns:
        A JSON-serializable pass report.

    Raises:
        ResearchPackageContractError: If any required manifest or artifact is
            missing or inconsistent.
    """

    root = Path(package_dir)
    errors: list[str] = []

    package_manifest_path = root / PACKAGE_MANIFEST_NAME
    package_manifest = _read_json_required(package_manifest_path, errors)
    root_manifest_path = root / ROOT_PROVENANCE_NAME
    root_manifest = _read_json_required(root_manifest_path, errors)

    if not package_manifest or not root_manifest:
        _raise_if_errors(errors)

    candidate_library_required = _candidate_library_required(package_manifest, require_candidate_library)
    package_manifest_check = _check_package_manifest(
        package_manifest,
        root,
        candidate_library_required=candidate_library_required,
        require_evidence_cache=require_evidence_cache,
        errors=errors,
    )
    required_artifacts_check = _check_required_artifacts(
        package_manifest,
        root,
        candidate_library_required=candidate_library_required,
        require_evidence_cache=require_evidence_cache,
        errors=errors,
    )
    root_provenance_check = _check_root_provenance(
        root_manifest,
        root,
        candidate_library_required=candidate_library_required,
        require_evidence_cache=require_evidence_cache,
        errors=errors,
    )
    evidence_cache_check = _check_evidence_cache(
        package_manifest,
        root,
        require_evidence_cache=require_evidence_cache,
        errors=errors,
    )
    scaffold_validation_check = _check_scaffold_validation(package_manifest, root, errors=errors)
    closed_loop_check = _check_closed_loop_iteration_packet(package_manifest, root, errors=errors)
    closure_readiness_check = _check_closure_readiness(
        package_manifest,
        root,
        require_closure_readiness=require_closure_readiness,
        errors=errors,
    )
    _raise_if_errors(errors)

    return {
        "status": "passed",
        "package_dir": str(root),
        "summary": {
            "dataset_id": package_manifest.get("dataset_id"),
            "evidence_mode": package_manifest.get("evidence_mode"),
            "input_scope": package_manifest.get("input_scope"),
            "publication_grade": package_manifest.get("publication_grade"),
            "verified_rows": package_manifest.get("verified_rows"),
            "quarantine_rows": package_manifest.get("quarantine_rows"),
            "ranked_candidates": package_manifest.get("ranked_candidates"),
            "candidate_library_rows": package_manifest.get("candidate_library_rows"),
            "candidate_library_required": candidate_library_required,
        },
        "checks": {
            "package_manifest": package_manifest_check,
            "required_artifacts": required_artifacts_check,
            "root_provenance": root_provenance_check,
            "evidence_cache": evidence_cache_check,
            "scaffold_validation": scaffold_validation_check,
            "closed_loop_iteration_packet": closed_loop_check,
            "closure_readiness": closure_readiness_check,
        },
    }


def _check_package_manifest(
    manifest: dict[str, Any],
    root: Path,
    *,
    candidate_library_required: bool,
    require_evidence_cache: bool,
    errors: list[str],
) -> dict[str, Any]:
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        errors.append(
            f"{PACKAGE_MANIFEST_NAME}: schema_version={manifest.get('schema_version')!r}; "
            f"expected {PACKAGE_SCHEMA_VERSION!r}"
        )
    outputs = _manifest_outputs(manifest)
    for key, kind in REQUIRED_PACKAGE_OUTPUTS.items():
        value = outputs.get(key)
        if not value:
            errors.append(f"{PACKAGE_MANIFEST_NAME}: outputs.{key} is required")
            continue
        _check_path(root, value, kind=kind, errors=errors, context=f"outputs.{key}")
    if candidate_library_required:
        value = outputs.get("candidate_library_dir")
        if not value:
            errors.append(f"{PACKAGE_MANIFEST_NAME}: outputs.candidate_library_dir is required")
        else:
            _check_path(root, value, kind="dir", errors=errors, context="outputs.candidate_library_dir")
    if require_evidence_cache:
        evidence_cache_dir = outputs.get("evidence_cache_dir")
        if evidence_cache_dir != "evidence_cache":
            errors.append(f"{PACKAGE_MANIFEST_NAME}: outputs.evidence_cache_dir must be 'evidence_cache'")
        else:
            _check_path(root, evidence_cache_dir, kind="dir", errors=errors, context="outputs.evidence_cache_dir")
        value = outputs.get("evidence_cache_manifest_json")
        if value != EVIDENCE_CACHE_MANIFEST_PATH:
            errors.append(
                f"{PACKAGE_MANIFEST_NAME}: outputs.evidence_cache_manifest_json must be "
                f"{EVIDENCE_CACHE_MANIFEST_PATH!r}"
            )
        else:
            _check_path(root, value, kind="file", errors=errors, context="outputs.evidence_cache_manifest_json")
    return {
        "status": "passed",
        "path": PACKAGE_MANIFEST_NAME,
        "schema_version": manifest.get("schema_version"),
        "output_keys": sorted(outputs),
    }


def _check_required_artifacts(
    manifest: dict[str, Any],
    root: Path,
    *,
    candidate_library_required: bool,
    require_evidence_cache: bool,
    errors: list[str],
) -> dict[str, Any]:
    required_paths = list(REQUIRED_ARTIFACT_FILES)
    outputs = _manifest_outputs(manifest)
    if candidate_library_required:
        candidate_dir = outputs.get("candidate_library_dir")
        if not candidate_dir:
            errors.append(f"{PACKAGE_MANIFEST_NAME}: outputs.candidate_library_dir is required")
        else:
            _check_path(root, candidate_dir, kind="dir", errors=errors, context="outputs.candidate_library_dir")
        required_paths.extend(CANDIDATE_LIBRARY_FILES)
    if require_evidence_cache:
        required_paths.append(EVIDENCE_CACHE_MANIFEST_PATH)

    for relative_path in required_paths:
        _check_path(root, relative_path, kind="file", errors=errors, context=relative_path)
    workflow_outputs = _check_workflow_outputs(manifest, root, errors)
    return {
        "status": "passed",
        "paths": sorted(required_paths),
        "workflow_outputs": workflow_outputs,
        "candidate_library_required": candidate_library_required,
    }


def _check_root_provenance(
    manifest: dict[str, Any],
    root: Path,
    *,
    candidate_library_required: bool,
    require_evidence_cache: bool,
    errors: list[str],
) -> dict[str, Any]:
    if manifest.get("schema_version") != ROOT_PROVENANCE_SCHEMA_VERSION:
        errors.append(
            f"{ROOT_PROVENANCE_NAME}: schema_version={manifest.get('schema_version')!r}; "
            f"expected {ROOT_PROVENANCE_SCHEMA_VERSION!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append(f"{ROOT_PROVENANCE_NAME}: artifacts must be an object")
        artifacts = {}

    required_ids: dict[str, set[str]] = {category: set(ids) for category, ids in REQUIRED_ROOT_IDS.items()}
    if candidate_library_required:
        for category, ids in CANDIDATE_LIBRARY_ROOT_IDS.items():
            required_ids.setdefault(category, set()).update(ids)
    if require_evidence_cache:
        for category, ids in EVIDENCE_CACHE_ROOT_IDS.items():
            required_ids.setdefault(category, set()).update(ids)

    artifact_ids_by_category: dict[str, list[str]] = {}
    for category, ids in required_ids.items():
        records = artifacts.get(category)
        if not isinstance(records, list):
            errors.append(f"{ROOT_PROVENANCE_NAME}: artifacts.{category} must be a list")
            artifact_ids_by_category[category] = []
            continue
        present = {str(item.get("id")) for item in records if isinstance(item, dict)}
        missing = ids - present
        if missing:
            errors.append(f"{ROOT_PROVENANCE_NAME}: artifacts.{category} missing ids {sorted(missing)}")
        records_by_id = {str(item.get("id")): item for item in records if isinstance(item, dict)}
        for artifact_id in sorted(ids & present):
            _check_root_record(
                root,
                records_by_id[artifact_id],
                errors=errors,
                context=f"{ROOT_PROVENANCE_NAME}: artifacts.{category}.{artifact_id}",
            )
        artifact_ids_by_category[category] = sorted(present)

    source_manifests = manifest.get("source_manifests")
    if not isinstance(source_manifests, dict) or "package_manifest_json" not in source_manifests:
        errors.append(f"{ROOT_PROVENANCE_NAME}: source_manifests.package_manifest_json is required")
    elif isinstance(source_manifests.get("package_manifest_json"), dict):
        _check_root_record(
            root,
            source_manifests["package_manifest_json"],
            errors=errors,
            context=f"{ROOT_PROVENANCE_NAME}: source_manifests.package_manifest_json",
            expected_path=root / PACKAGE_MANIFEST_NAME,
        )
    else:
        errors.append(f"{ROOT_PROVENANCE_NAME}: source_manifests.package_manifest_json must be an object")

    return {
        "status": "passed",
        "path": ROOT_PROVENANCE_NAME,
        "schema_version": manifest.get("schema_version"),
        "artifact_ids": artifact_ids_by_category,
    }


def _check_evidence_cache(
    manifest: dict[str, Any],
    root: Path,
    *,
    require_evidence_cache: bool,
    errors: list[str],
) -> dict[str, Any]:
    if not require_evidence_cache:
        return {"status": "skipped", "reason": "require_evidence_cache=false"}

    if manifest.get("evidence_mode") != "external-cached":
        errors.append("evidence cache was required, but package evidence_mode is not external-cached")
        return {"status": "failed", "reason": "evidence_mode is not external-cached"}

    runner_args = manifest.get("runner_args")
    if not isinstance(runner_args, dict):
        runner_args = {}
    cache_dir_value = runner_args.get("cache_dir")
    if not cache_dir_value:
        errors.append(f"{PACKAGE_MANIFEST_NAME}: runner_args.cache_dir is required for evidence cache verification")
        return {"status": "failed", "reason": "runner_args.cache_dir missing"}

    cache_dir = _resolve_cache_dir(root, str(cache_dir_value))
    manifest_path = root / EVIDENCE_CACHE_MANIFEST_PATH
    evidence_manifest = _read_json_artifact(
        manifest_path,
        root,
        context=EVIDENCE_CACHE_MANIFEST_PATH,
        errors=errors,
    )
    if not isinstance(evidence_manifest, dict):
        return {"status": "failed", "reason": "evidence cache manifest missing or invalid"}
    if evidence_manifest.get("schema_version") != EVIDENCE_CACHE_MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"{EVIDENCE_CACHE_MANIFEST_PATH}: schema_version={evidence_manifest.get('schema_version')!r}; "
            f"expected {EVIDENCE_CACHE_MANIFEST_SCHEMA_VERSION!r}"
        )
    manifest_cache_dir = evidence_manifest.get("cache_dir")
    if not isinstance(manifest_cache_dir, str) or not manifest_cache_dir:
        errors.append(f"{EVIDENCE_CACHE_MANIFEST_PATH}: cache_dir is required")
    elif _resolve_external_path(root, manifest_cache_dir).resolve() != cache_dir.resolve():
        errors.append(
            f"{EVIDENCE_CACHE_MANIFEST_PATH}: cache_dir={manifest_cache_dir!r} "
            f"does not match runner_args.cache_dir={cache_dir_value!r}"
        )
    manifest_cache_files = evidence_manifest.get("cache_files")
    if not isinstance(manifest_cache_files, dict):
        errors.append(f"{EVIDENCE_CACHE_MANIFEST_PATH}: cache_files must be an object")
        manifest_cache_files = {}

    cache_files: dict[str, dict[str, Any]] = {}
    for cache_key, filename in EVIDENCE_CACHE_FILES:
        path = cache_dir / filename
        if not path.is_file():
            errors.append(f"evidence_cache/{filename}: missing at {path}")
            continue
        cache_files[cache_key] = _check_cache_file_manifest(
            cache_key,
            filename,
            path,
            manifest_cache_files.get(cache_key),
            root=root,
            errors=errors,
        )
    return {
        "status": "passed",
        "cache_dir": str(cache_dir),
        "manifest_path": _display_path(manifest_path, root),
        "cache_files": cache_files,
    }


def _check_closed_loop_iteration_packet(
    manifest: dict[str, Any],
    root: Path,
    *,
    errors: list[str],
) -> dict[str, Any]:
    outputs = _manifest_outputs(manifest)
    iteration_dir_value = outputs.get("closed_loop_iteration_dir", "closed_loop/iteration_000")
    ranked_candidates_value = outputs.get(
        "ranked_candidates_csv",
        "verified_discovery/discovery/ranked_candidates.csv",
    )
    runner_args = manifest.get("runner_args")
    if not isinstance(runner_args, dict):
        runner_args = {}
    top_k = int(runner_args.get("iteration_top_k") or runner_args.get("top_k") or 10)
    try:
        report = verify_iteration_packet(
            _resolve_path(root, str(iteration_dir_value)),
            ranked_candidates_csv=_resolve_path(root, str(ranked_candidates_value)),
            top_k=top_k,
        )
    except (ClosedLoopPacketError, OSError, ValueError) as exc:
        errors.append(f"closed_loop iteration packet failed: {exc}")
        return {"status": "failed", "error": str(exc)}
    return report


def _check_scaffold_validation(
    manifest: dict[str, Any],
    root: Path,
    *,
    errors: list[str],
) -> dict[str, Any]:
    outputs = _manifest_outputs(manifest)
    metrics_value = outputs.get(
        "model_scaffold_validation_metrics_json",
        "verified_discovery/model/model_scaffold_validation_metrics.json",
    )
    predictions_value = outputs.get(
        "model_scaffold_validation_predictions_csv",
        "verified_discovery/model/model_scaffold_validation_predictions.csv",
    )
    metrics_path = _resolve_path(root, str(metrics_value))
    predictions_path = _resolve_path(root, str(predictions_value))
    metrics = _read_json_artifact(
        metrics_path,
        root,
        context="verified_discovery/model/model_scaffold_validation_metrics.json",
        errors=errors,
    )
    if not isinstance(metrics, dict):
        return {"status": "failed", "reason": "scaffold validation metrics missing or invalid"}

    if metrics.get("schema_version") != "model-scaffold-validation-metrics-v1":
        errors.append(
            "verified_discovery/model/model_scaffold_validation_metrics.json: "
            f"schema_version={metrics.get('schema_version')!r}; expected 'model-scaffold-validation-metrics-v1'"
        )
    if metrics.get("validation_scope") != "chemical_group_holdout_verified_rows":
        errors.append(
            "verified_discovery/model/model_scaffold_validation_metrics.json: "
            "validation_scope must be 'chemical_group_holdout_verified_rows'"
        )
    if metrics.get("external_validation") is not False:
        errors.append(
            "verified_discovery/model/model_scaffold_validation_metrics.json: external_validation must be false"
        )
    if metrics.get("random_holdout") is not False:
        errors.append("verified_discovery/model/model_scaffold_validation_metrics.json: random_holdout must be false")
    if metrics.get("chemical_generalization_diagnostic") is not True:
        errors.append(
            "verified_discovery/model/model_scaffold_validation_metrics.json: "
            "chemical_generalization_diagnostic must be true"
        )
    if metrics.get("chemical_group_column") != "chemical_group_key":
        errors.append(
            "verified_discovery/model/model_scaffold_validation_metrics.json: "
            "chemical_group_column must be 'chemical_group_key'"
        )

    train_groups = [str(group) for group in metrics.get("train_groups", [])]
    validation_groups = [str(group) for group in metrics.get("validation_groups", [])]
    overlap = sorted(set(train_groups) & set(validation_groups))
    declared_overlap = [str(group) for group in metrics.get("overlap_groups", [])]
    if overlap or declared_overlap:
        errors.append(
            "verified_discovery/model/model_scaffold_validation_metrics.json: "
            f"train/validation groups overlap ({sorted(set(overlap) | set(declared_overlap))})"
        )
    status = metrics.get("status")
    if status == "passed":
        if not train_groups or not validation_groups:
            errors.append(
                "verified_discovery/model/model_scaffold_validation_metrics.json: "
                "passed scaffold validation requires non-empty train_groups and validation_groups"
            )
        _check_scaffold_predictions(predictions_path, root, metrics=metrics, errors=errors)
    elif status == "skipped":
        if not metrics.get("skip_reason"):
            errors.append(
                "verified_discovery/model/model_scaffold_validation_metrics.json: "
                "skipped scaffold validation requires skip_reason"
            )
    else:
        errors.append(
            "verified_discovery/model/model_scaffold_validation_metrics.json: "
            f"status must be 'passed' or 'skipped'; observed {status!r}"
        )
    return {
        "status": status,
        "path": _display_path(metrics_path, root),
        "skip_reason": metrics.get("skip_reason"),
        "validation_rows": metrics.get("validation_rows"),
        "train_groups": train_groups,
        "validation_groups": validation_groups,
    }


def _check_scaffold_predictions(
    path: Path,
    root: Path,
    *,
    metrics: dict[str, Any],
    errors: list[str],
) -> None:
    expected = [
        "record_id",
        "smiles",
        "chemical_group_key",
        "chemical_group_source",
        "y_true",
        "y_pred",
        "residual",
        "split",
    ]
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames
            rows = list(reader)
    except OSError as exc:
        errors.append(f"{_display_path(path, root)}: cannot read scaffold predictions CSV ({exc})")
        return
    if header != expected:
        errors.append(f"{_display_path(path, root)}: expected CSV header {expected}, observed {header}")
        return

    relative = _display_path(path, root)
    expected_rows = int(metrics.get("validation_rows") or 0)
    if len(rows) != expected_rows:
        errors.append(f"{relative}: scaffold predictions row count {len(rows)} != validation_rows {expected_rows}")
    train_groups = {str(group) for group in metrics.get("train_groups", [])}
    validation_groups = {str(group) for group in metrics.get("validation_groups", [])}
    record_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        prefix = f"{relative}: scaffold predictions row {row_number}"
        record_id = str(row.get("record_id", ""))
        if record_id in record_ids:
            errors.append(f"{prefix}: duplicate record_id {record_id!r}")
        record_ids.add(record_id)
        if row.get("split") != "validation":
            errors.append(f"{prefix}: split must be 'validation'")
        group_key = str(row.get("chemical_group_key", ""))
        if group_key in train_groups:
            errors.append(f"{prefix}: chemical_group_key {group_key!r} is a train group")
        if group_key not in validation_groups:
            errors.append(f"{prefix}: chemical_group_key {group_key!r} is not in validation_groups")
        y_true = _parse_float(row.get("y_true"))
        y_pred = _parse_float(row.get("y_pred"))
        residual = _parse_float(row.get("residual"))
        if y_true is None or y_pred is None or residual is None:
            errors.append(f"{prefix}: y_true, y_pred, and residual must be finite numbers")
            continue
        if not math.isclose(residual, y_pred - y_true, rel_tol=1e-7, abs_tol=1e-7):
            errors.append(f"{prefix}: residual does not equal y_pred - y_true")


def _parse_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _check_closure_readiness(
    manifest: dict[str, Any],
    root: Path,
    *,
    require_closure_readiness: bool,
    errors: list[str],
) -> dict[str, Any]:
    outputs = _manifest_outputs(manifest)
    value = outputs.get("closure_readiness_report_json", CLOSURE_READINESS_REPORT_PATH)
    report_path = _resolve_path(root, str(value))
    report = _read_json_artifact(
        report_path,
        root,
        context=CLOSURE_READINESS_REPORT_PATH,
        errors=errors,
    )
    if not isinstance(report, dict):
        return {"status": "failed", "reason": "closure readiness report missing or invalid"}
    if report.get("schema_version") != "closure-readiness-report-v1":
        errors.append(
            f"{CLOSURE_READINESS_REPORT_PATH}: schema_version={report.get('schema_version')!r}; "
            "expected 'closure-readiness-report-v1'"
        )
    if report.get("dataset_id") != manifest.get("dataset_id"):
        errors.append(
            f"{CLOSURE_READINESS_REPORT_PATH}: dataset_id={report.get('dataset_id')!r} "
            f"does not match package dataset_id={manifest.get('dataset_id')!r}"
        )
    if report.get("publication_grade") is not True:
        reasons = report.get("publication_grade_false_reasons")
        if not isinstance(reasons, list) or not reasons:
            errors.append(f"{CLOSURE_READINESS_REPORT_PATH}: publication_grade_false_reasons must be non-empty")
    gates = report.get("gates")
    if not isinstance(gates, dict):
        errors.append(f"{CLOSURE_READINESS_REPORT_PATH}: gates must be an object")
        gates = {}
    chemical_gate = gates.get("chemical_generalization_validation")
    if not isinstance(chemical_gate, dict):
        errors.append(f"{CLOSURE_READINESS_REPORT_PATH}: gates.chemical_generalization_validation is required")
    elif not isinstance(chemical_gate.get("passed"), bool):
        errors.append(f"{CLOSURE_READINESS_REPORT_PATH}: gates.chemical_generalization_validation.passed must be boolean")
    overall_ready = report.get("overall_ready")
    if require_closure_readiness and overall_ready is not True:
        errors.append(
            "closure readiness required overall_ready=true; "
            f"observed overall_ready={overall_ready!r}"
        )
    return {
        "status": "passed",
        "path": _display_path(report_path, root),
        "overall_ready": overall_ready,
        "blocking_reasons": report.get("blocking_reasons", []),
    }


def _candidate_library_required(manifest: dict[str, Any], require_candidate_library: bool) -> bool:
    outputs = _manifest_outputs(manifest)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
    runner_args = manifest.get("runner_args")
    if not isinstance(runner_args, dict):
        runner_args = {}
    return (
        require_candidate_library
        or outputs.get("candidate_library_dir") is not None
        or manifest.get("candidate_library_rows") is not None
        or inputs.get("candidate_source") is not None
        or runner_args.get("candidate_source") is not None
    )


def _manifest_outputs(manifest: dict[str, Any]) -> dict[str, Any]:
    outputs = manifest.get("outputs")
    return outputs if isinstance(outputs, dict) else {}


def _read_json_required(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"{path.name}: missing")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{path.name}: invalid JSON ({exc})")
        return {}


def _check_path(root: Path, value: str | Path, *, kind: str, errors: list[str], context: str) -> None:
    path = _resolve_path(root, value)
    relative = _display_path(path, root)
    if kind == "dir":
        if not path.is_dir():
            errors.append(f"{context}: missing directory {relative}")
        return
    if not path.is_file():
        errors.append(f"{context}: missing file {relative}")
        return
    _check_artifact_payload(root, path, context=context, errors=errors)


def _check_workflow_outputs(manifest: dict[str, Any], root: Path, errors: list[str]) -> dict[str, Any]:
    outputs = _manifest_outputs(manifest)
    verified_discovery_dir = outputs.get("verified_discovery_dir")
    if not verified_discovery_dir:
        errors.append(f"{PACKAGE_MANIFEST_NAME}: outputs.verified_discovery_dir is required for workflow output checks")
        return {"status": "failed", "output_keys": []}

    discovery_root = _resolve_path(root, verified_discovery_dir)
    workflow_path = discovery_root / "workflow_manifest.json"
    workflow_manifest = _read_json_artifact(workflow_path, root, context="verified_discovery/workflow_manifest.json", errors=errors)
    workflow_outputs = workflow_manifest.get("outputs") if isinstance(workflow_manifest, dict) else None
    if not isinstance(workflow_outputs, dict):
        errors.append("verified_discovery/workflow_manifest.json: outputs must be an object")
        return {"status": "failed", "output_keys": []}

    for key in sorted(REQUIRED_WORKFLOW_OUTPUTS):
        value = workflow_outputs.get(key)
        if not value:
            errors.append(f"verified_discovery/workflow_manifest.json: outputs.{key} is required")
            continue
        _check_path(
            Path("."),
            discovery_root / str(value),
            kind="file",
            errors=errors,
            context=f"workflow.outputs.{key}",
        )
    return {"status": "passed", "output_keys": sorted(workflow_outputs)}


def _check_artifact_payload(root: Path, path: Path, *, context: str, errors: list[str]) -> None:
    relative = _display_path(path, root)
    if path.suffix == ".json":
        payload = _read_json_artifact(path, root, context=relative, errors=errors)
        if payload is None:
            return
        expected_fields = JSON_FIELD_REQUIREMENTS.get(relative)
        if expected_fields and isinstance(payload, dict):
            missing = [field for field in expected_fields if field not in payload]
            if missing:
                errors.append(f"{relative}: missing JSON fields {missing}")
        elif expected_fields:
            errors.append(f"{relative}: JSON payload must be an object")
        elif relative == "report_bundle/main_text/claim_ledger.json" and not isinstance(payload, list):
            errors.append(f"{relative}: JSON payload must be a claim list")
        return

    if path.suffix in {".csv", ".md"} and path.stat().st_size == 0:
        errors.append(f"{context}: empty artifact {relative}")


def _read_json_artifact(path: Path, root: Path, *, context: str, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{context}: invalid JSON ({exc})")
    except OSError as exc:
        errors.append(f"{context}: cannot read JSON ({exc})")
    return None


def _check_cache_file_manifest(
    cache_key: str,
    filename: str,
    path: Path,
    manifest_record: Any,
    *,
    root: Path,
    errors: list[str],
) -> dict[str, Any]:
    context = f"{EVIDENCE_CACHE_MANIFEST_PATH}: cache_files.{cache_key}"
    if not isinstance(manifest_record, dict):
        errors.append(f"{context} is required")
        manifest_record = {}

    payload = _read_json_artifact(path, path.parent, context=str(path), errors=errors)
    entry_count = len(payload) if isinstance(payload, dict) else None
    if payload is not None and entry_count is None:
        errors.append(f"{path}: JSON payload must be an object")

    size_bytes = path.stat().st_size
    sha256 = _sha256(path)
    expected_size = manifest_record.get("size_bytes")
    expected_sha256 = manifest_record.get("sha256")
    expected_entry_count = manifest_record.get("entry_count")
    expected_exists = manifest_record.get("exists")
    expected_filename = manifest_record.get("filename")
    expected_path = manifest_record.get("path")

    if expected_exists is not True:
        errors.append(f"{context}.exists must be true")
    if expected_filename != filename:
        errors.append(f"{context}.filename={expected_filename!r} does not match {filename!r}")
    if not isinstance(expected_path, str) or not expected_path:
        errors.append(f"{context}.path is required")
    elif _resolve_external_path(root, expected_path).resolve() != path.resolve():
        errors.append(f"{context}.path={expected_path!r} does not match expected {path}")
    if expected_size != size_bytes:
        errors.append(f"{context}.size_bytes={expected_size!r} does not match disk size {size_bytes}")
    if expected_sha256 != sha256:
        errors.append(f"{context}.sha256 does not match disk content")
    if expected_entry_count != entry_count:
        errors.append(f"{context}.entry_count={expected_entry_count!r} does not match JSON object count {entry_count}")

    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "entry_count": entry_count,
    }


def _check_root_record(
    root: Path,
    record: dict[str, Any],
    *,
    errors: list[str],
    context: str,
    expected_path: Path | None = None,
) -> None:
    if record.get("exists") is not True:
        errors.append(f"{context}: exists must be true")
    path_value = record.get("path")
    if not path_value:
        errors.append(f"{context}: path is required")
        return

    path = _resolve_path(root, str(path_value))
    if expected_path is not None and path.resolve() != expected_path.resolve():
        errors.append(
            f"{context}: path {path_value} does not match expected {_display_path(expected_path, root)}"
        )
    if not path.is_file():
        errors.append(f"{context}: missing file {_display_path(path, root)}")
        return

    size_bytes = record.get("size_bytes")
    if isinstance(size_bytes, int) and size_bytes != path.stat().st_size:
        errors.append(f"{context}: size_bytes={size_bytes} does not match disk size {path.stat().st_size}")
    sha256_16 = record.get("sha256_16")
    if isinstance(sha256_16, str) and sha256_16 != _sha256_16(path):
        errors.append(f"{context}: sha256_16 does not match disk content")


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _resolve_cache_dir(root: Path, value: str) -> Path:
    return _resolve_external_path(root, value)


def _resolve_external_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    return root.parent / path


def _sha256_16(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()[:16]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _raise_if_errors(errors: list[str]) -> None:
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ResearchPackageContractError(f"Research package artifact contract failed:\n{joined}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, help="Directory emitted by run_research_package.")
    parser.add_argument("--require-candidate-library", action="store_true")
    parser.add_argument("--require-evidence-cache", action="store_true")
    parser.add_argument("--require-closure-readiness", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_research_package(
            args.package_dir,
            require_candidate_library=args.require_candidate_library,
            require_evidence_cache=args.require_evidence_cache,
            require_closure_readiness=args.require_closure_readiness,
        )
    except ResearchPackageContractError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
