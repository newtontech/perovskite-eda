"""Verify that a generated research package satisfies the artifact contract.

The verifier is intentionally read-only. It checks that the package manifest,
root provenance manifest, and the expected scientific outputs agree with the
files on disk before a package is treated as a complete research deliverable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PACKAGE_MANIFEST_NAME = "package_manifest.json"
ROOT_PROVENANCE_NAME = "provenance_manifest.json"
PACKAGE_SCHEMA_VERSION = "research-package-manifest-v1"
ROOT_PROVENANCE_SCHEMA_VERSION = "root-provenance-manifest-v1"

REQUIRED_PACKAGE_OUTPUTS = {
    "verified_discovery_dir": "dir",
    "source_completeness_dir": "dir",
    "report_dir": "dir",
    "main_text_report_md": "file",
    "supporting_information_md": "file",
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
)

CANDIDATE_LIBRARY_FILES = (
    "candidate_library/candidate_library.csv",
    "candidate_library/source_summary.json",
    "candidate_library/provenance.json",
)

EVIDENCE_CACHE_FILES = (
    "reference_cache.json",
    "molecule_cache.json",
)

REQUIRED_ROOT_IDS = {
    "dataset": {
        "verified_train_csv",
        "quarantine_csv",
        "candidate_pool_csv",
        "doi_manifest_json",
        "provenance_json",
        "data_audit_report_md",
    },
    "model": {"model_metrics_json", "model_manifest_json"},
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
    "package": {"package_manifest_json"},
}

CANDIDATE_LIBRARY_ROOT_IDS = {
    "discovery": {"candidate_library:candidate_library_csv"},
    "audit": {
        "candidate_library:source_summary_json",
        "candidate_library:provenance_json",
    },
}


class ResearchPackageContractError(RuntimeError):
    """Raised when a research package violates the artifact contract."""


def verify_research_package(
    package_dir: str | Path,
    *,
    require_candidate_library: bool = False,
    require_evidence_cache: bool = False,
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

    package_manifest_check = _check_package_manifest(package_manifest, root, errors)
    candidate_library_required = _candidate_library_required(package_manifest, require_candidate_library)
    required_artifacts_check = _check_required_artifacts(
        package_manifest,
        root,
        candidate_library_required=candidate_library_required,
        errors=errors,
    )
    root_provenance_check = _check_root_provenance(
        root_manifest,
        candidate_library_required=candidate_library_required,
        errors=errors,
    )
    evidence_cache_check = _check_evidence_cache(
        package_manifest,
        root,
        require_evidence_cache=require_evidence_cache,
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
        },
    }


def _check_package_manifest(
    manifest: dict[str, Any],
    root: Path,
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

    for relative_path in required_paths:
        _check_path(root, relative_path, kind="file", errors=errors, context=relative_path)
    return {
        "status": "passed",
        "paths": sorted(required_paths),
        "candidate_library_required": candidate_library_required,
    }


def _check_root_provenance(
    manifest: dict[str, Any],
    *,
    candidate_library_required: bool,
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
        artifact_ids_by_category[category] = sorted(present)

    source_manifests = manifest.get("source_manifests")
    if not isinstance(source_manifests, dict) or "package_manifest_json" not in source_manifests:
        errors.append(f"{ROOT_PROVENANCE_NAME}: source_manifests.package_manifest_json is required")

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
    paths: list[str] = []
    for filename in EVIDENCE_CACHE_FILES:
        path = cache_dir / filename
        if not path.is_file():
            errors.append(f"evidence_cache/{filename}: missing at {path}")
        paths.append(str(path))
    return {"status": "passed", "cache_dir": str(cache_dir), "paths": paths}


def _candidate_library_required(manifest: dict[str, Any], require_candidate_library: bool) -> bool:
    outputs = _manifest_outputs(manifest)
    return (
        require_candidate_library
        or outputs.get("candidate_library_dir") is not None
        or manifest.get("candidate_library_rows") is not None
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


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _resolve_cache_dir(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    return root.parent / path


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_research_package(
            args.package_dir,
            require_candidate_library=args.require_candidate_library,
            require_evidence_cache=args.require_evidence_cache,
        )
    except ResearchPackageContractError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
