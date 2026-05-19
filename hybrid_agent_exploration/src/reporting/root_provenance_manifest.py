"""Generate the root provenance manifest for a report bundle.

The manifest intentionally records lightweight file facts only. It indexes the
verified discovery workflow outputs alongside report and SI sidecars without
embedding source data or model artifacts into the manifest itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_NAME = "provenance_manifest.json"
SCHEMA_VERSION = "root-provenance-manifest-v1"
HASH_BYTES_LIMIT = 1024 * 1024
ARTIFACT_CATEGORIES = ("dataset", "model", "discovery", "report", "SI", "claim", "review", "audit")


def generate_root_provenance_manifest(
    verified_discovery_artifact_dir: str | Path,
    report_dir: str | Path,
    si_dir: str | Path,
    *,
    candidate_library_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and write a root-level provenance manifest.

    Args:
        verified_discovery_artifact_dir: Directory containing workflow_manifest.json.
        report_dir: Directory containing main report artifacts.
        si_dir: Directory containing supplementary information artifacts.
        candidate_library_dir: Optional directory containing candidate_library.csv,
            source_summary.json, and provenance.json from CandidateLibraryBuilder.
        output_path: Optional explicit path. Defaults to the common parent of
            report_dir and si_dir plus ``provenance_manifest.json``.

    Returns:
        The JSON-serializable manifest that was written.
    """

    discovery_root = Path(verified_discovery_artifact_dir)
    report_root = Path(report_dir)
    si_root = Path(si_dir)
    candidate_root = Path(candidate_library_dir) if candidate_library_dir is not None else None
    output = Path(output_path) if output_path is not None else _default_output_path(report_root, si_root)
    root_dir = output.parent

    workflow_manifest_path = discovery_root / "workflow_manifest.json"
    workflow_manifest = _read_json(workflow_manifest_path)

    artifacts: dict[str, list[dict[str, Any]]] = {category: [] for category in ARTIFACT_CATEGORIES}
    for artifact_id, relative_path in _workflow_outputs(workflow_manifest).items():
        category = _category_for_workflow_output(artifact_id)
        if category is None:
            continue
        artifacts[category].append(
            _artifact_record(
                artifact_id,
                _resolve_artifact_path(discovery_root, relative_path),
                root_dir=root_dir,
                source_root=discovery_root,
                declared_in="workflow_manifest.outputs",
            )
        )

    for path in _iter_files(report_root):
        category = _category_for_report_sidecar(path, default="report")
        artifacts[category].append(
            _artifact_record(
                _artifact_id_from_path(path),
                path,
                root_dir=root_dir,
                source_root=report_root,
                declared_in="report_dir",
            )
        )

    for path in _iter_files(si_root):
        category = _category_for_report_sidecar(path, default="SI")
        artifacts[category].append(
            _artifact_record(
                _artifact_id_from_path(path),
                path,
                root_dir=root_dir,
                source_root=si_root,
                declared_in="si_dir",
            )
        )

    if candidate_root is not None:
        for path in _iter_files(candidate_root):
            artifact_id = f"candidate_library:{_artifact_id_from_path(path)}"
            category = _category_for_candidate_library(path)
            artifacts[category].append(
                _artifact_record(
                    artifact_id,
                    path,
                    root_dir=root_dir,
                    source_root=candidate_root,
                    declared_in="candidate_library_dir",
                )
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "dataset_id": workflow_manifest.get("dataset_id"),
        "artifact_policy": workflow_manifest.get("artifact_policy"),
        "strict_verified_training_only": bool(workflow_manifest.get("strict_verified_training_only")),
        "verified_candidate_discovery_only": bool(workflow_manifest.get("verified_candidate_discovery_only")),
        "hash_policy": {
            "algorithm": "sha256",
            "digest": "first_16_hex_chars",
            "max_bytes_hashed_per_file": HASH_BYTES_LIMIT,
        },
        "roots": {
            "verified_discovery_artifact_dir": _display_path(discovery_root, root_dir),
            "report_dir": _display_path(report_root, root_dir),
            "si_dir": _display_path(si_root, root_dir),
        },
        "source_manifests": {
            "workflow_manifest_json": _file_facts(workflow_manifest_path, root_dir=root_dir),
        },
        "artifacts": {category: sorted(records, key=lambda item: item["id"]) for category, records in artifacts.items()},
    }
    if candidate_root is not None:
        manifest["roots"]["candidate_library_dir"] = _display_path(candidate_root, root_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ad hoc manifest generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-discovery-artifact-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--si-dir", required=True)
    parser.add_argument("--candidate-library-dir")
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)
    manifest = generate_root_provenance_manifest(
        args.verified_discovery_artifact_dir,
        args.report_dir,
        args.si_dir,
        candidate_library_dir=args.candidate_library_dir,
        output_path=args.output_path,
    )
    output_path = args.output_path or _default_output_path(Path(args.report_dir), Path(args.si_dir))
    print(f"[root-provenance] wrote {output_path} for dataset_id={manifest.get('dataset_id')}")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Verified discovery workflow manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _workflow_outputs(workflow_manifest: dict[str, Any]) -> dict[str, str]:
    outputs = workflow_manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        return {}
    return {str(key): str(value) for key, value in outputs.items()}


def _category_for_workflow_output(artifact_id: str) -> str | None:
    if artifact_id.startswith(("verified_", "quarantine", "candidate_pool", "doi_", "provenance", "data_")):
        return "dataset"
    if artifact_id.startswith("model_"):
        return "model"
    if artifact_id.startswith(("ranked_", "discovery_")):
        return "audit" if "audit" in artifact_id else "discovery"
    return None


def _category_for_report_sidecar(path: Path, *, default: str) -> str:
    name = path.name.lower()
    if "claim" in name:
        return "claim"
    if "review" in name:
        return "review"
    if "audit" in name:
        return "audit"
    return default


def _category_for_candidate_library(path: Path) -> str:
    name = path.name.lower()
    if name == "candidate_library.csv":
        return "discovery"
    if "provenance" in name or "summary" in name:
        return "audit"
    return "discovery"


def _artifact_record(
    artifact_id: str,
    path: Path,
    *,
    root_dir: Path,
    source_root: Path,
    declared_in: str,
) -> dict[str, Any]:
    facts = _file_facts(path, root_dir=root_dir)
    facts.update(
        {
            "id": artifact_id,
            "source_root": _display_path(source_root, root_dir),
            "declared_in": declared_in,
        }
    )
    return facts


def _file_facts(path: Path, *, root_dir: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": _display_path(path, root_dir),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "sha256_16": _sha256_16(path) if exists and path.is_file() else None,
    }


def _sha256_16(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(HASH_BYTES_LIMIT))
    return digest.hexdigest()[:16]


def _resolve_artifact_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name != MANIFEST_NAME)


def _artifact_id_from_path(path: Path) -> str:
    return path.name.replace(".", "_").replace("-", "_")


def _default_output_path(report_dir: Path, si_dir: Path) -> Path:
    common_root = Path(os.path.commonpath([report_dir.resolve(), si_dir.resolve()]))
    return common_root / MANIFEST_NAME


def _display_path(path: Path, root_dir: Path) -> str:
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
