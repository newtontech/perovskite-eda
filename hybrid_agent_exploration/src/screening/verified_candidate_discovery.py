"""Verified-only candidate discovery outputs.

Candidate discovery must not silently mix quarantined or unverifiable rows into
screening results. This module enforces `verification_status == "verified"`
before ranking and writes a small, Git-trackable discovery bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


FeatureFn = Callable[[pd.Series], Any]
UncertaintyFn = Callable[[Any, np.ndarray], Any]


class UnverifiedCandidatePoolError(ValueError):
    """Raised when a candidate pool contains non-verified rows."""


class CandidateLibraryContractError(ValueError):
    """Raised when a candidate pool is not provenance-complete."""


CANDIDATE_LIBRARY_CONTRACT_VERSION = "candidate-library-v1"
CANDIDATE_LIBRARY_REQUIRED_COLUMNS = (
    "candidate_id",
    "smiles",
    "source_name",
    "source_url",
    "availability_status",
    "synthesis_status",
    "safety_status",
    "verification_status",
    "verification_sources",
)


@dataclass(frozen=True)
class CandidateDiscoveryArtifacts:
    """Paths and counts emitted by a verified candidate discovery run."""

    output_dir: Path
    ranked_candidates_csv: Path
    discovery_manifest_json: Path
    audit_report_md: Path
    input_count: int
    ranked_count: int


class VerifiedCandidateDiscovery:
    """Rank candidates only after strict verification status checks pass."""

    def __init__(self, *, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def discover(
        self,
        candidate_pool: pd.DataFrame | str | Path,
        *,
        model: Any,
        feature_fn: FeatureFn,
        top_k: int = 100,
        dataset_id: str,
        uncertainty_fn: UncertaintyFn | None = None,
    ) -> CandidateDiscoveryArtifacts:
        """Rank a verified candidate pool and write discovery artifacts."""

        df = _load_candidate_pool(candidate_pool)
        validate_candidate_library_contract(df)
        _require_verified(df)
        ranked = _rank_candidates(
            df,
            model=model,
            feature_fn=feature_fn,
            top_k=top_k,
            uncertainty_fn=uncertainty_fn,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = CandidateDiscoveryArtifacts(
            output_dir=self.output_dir,
            ranked_candidates_csv=self.output_dir / "ranked_candidates.csv",
            discovery_manifest_json=self.output_dir / "candidate_discovery_manifest.json",
            audit_report_md=self.output_dir / "candidate_discovery_audit.md",
            input_count=len(df),
            ranked_count=len(ranked),
        )
        ranked.to_csv(artifacts.ranked_candidates_csv, index=False)
        _write_json(_manifest(dataset_id, artifacts, df, ranked, top_k, model), artifacts.discovery_manifest_json)
        artifacts.audit_report_md.write_text(
            _audit_report(dataset_id, artifacts, ranked),
            encoding="utf-8",
        )
        return artifacts


def _load_candidate_pool(candidate_pool: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(candidate_pool, pd.DataFrame):
        return candidate_pool.copy()
    path = Path(candidate_pool)
    if not path.exists():
        raise FileNotFoundError(f"Candidate pool not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported candidate pool format: {path.suffix}")


def validate_candidate_library_contract(df: pd.DataFrame) -> None:
    """Ensure candidate rows carry enough identity and provenance for ranking."""

    missing = [column for column in CANDIDATE_LIBRARY_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise CandidateLibraryContractError(
            "Candidate library contract missing required columns: " + ", ".join(missing)
        )

    empty_errors: list[str] = []
    for column in CANDIDATE_LIBRARY_REQUIRED_COLUMNS:
        empty = df[column].map(_text) == ""
        if empty.any():
            empty_errors.append(f"{column} empty for rows: {', '.join(_row_identifiers(df[empty]))}")
    if empty_errors:
        raise CandidateLibraryContractError("; ".join(empty_errors))

    source_errors: list[str] = []
    identity_errors: list[str] = []
    for idx, row in df.iterrows():
        sources = _parse_verification_sources(row.get("verification_sources"))
        if not sources:
            identifier = _row_identifiers(df.loc[[idx]])[0]
            source_errors.append(f"{identifier} has empty verification_sources")
            continue
        has_identity = _text(row.get("pubchem_id")) or _text(row.get("cas_number"))
        has_molecule_source = any(_text(source.get("kind")) == "molecule" for source in sources)
        if not has_identity and not has_molecule_source:
            identifier = _row_identifiers(df.loc[[idx]])[0]
            identity_errors.append(f"{identifier} lacks molecule identity")
    if source_errors:
        raise CandidateLibraryContractError("; ".join(source_errors))
    if identity_errors:
        raise CandidateLibraryContractError("; ".join(identity_errors))


def _require_verified(df: pd.DataFrame) -> None:
    if "verification_status" not in df.columns:
        raise UnverifiedCandidatePoolError(
            "Candidate discovery requires verification_status=verified for every row; "
            "missing column verification_status."
        )
    status = df["verification_status"].fillna("").astype(str)
    bad = df[status != "verified"]
    if bad.empty:
        return
    identifiers = _row_identifiers(bad)
    raise UnverifiedCandidatePoolError(
        "Candidate discovery requires verification_status=verified for every row; "
        f"blocked rows: {', '.join(identifiers)}"
    )


def _row_identifiers(df: pd.DataFrame, limit: int = 10) -> list[str]:
    if "candidate_id" in df.columns:
        values = df["candidate_id"].fillna("").astype(str)
        ids = [value.strip() for value in values.tolist() if value.strip()]
        if ids:
            return ids[:limit]
    if "record_id" in df.columns:
        values = df["record_id"].fillna("").astype(str)
        ids = [value.strip() for value in values.tolist() if value.strip()]
    else:
        ids = [f"index:{idx}" for idx in df.index.tolist()]
    return ids[:limit]


def _rank_candidates(
    df: pd.DataFrame,
    *,
    model: Any,
    feature_fn: FeatureFn,
    top_k: int,
    uncertainty_fn: UncertaintyFn | None,
) -> pd.DataFrame:
    if "smiles" not in df.columns:
        raise ValueError("Candidate pool must contain a smiles column.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    features = feature_fn(df["smiles"])
    X = _as_feature_array(features)
    predictions = np.asarray(model.predict(X), dtype=float)
    if len(predictions) != len(df):
        raise ValueError("Model prediction length does not match candidate pool length.")

    ranked = df.copy()
    ranked["predicted_delta_pce"] = predictions
    if uncertainty_fn is not None:
        ranked["uncertainty"] = np.asarray(uncertainty_fn(model, X), dtype=float)
    ranked = ranked.sort_values("predicted_delta_pce", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked.head(top_k).reset_index(drop=True)


def _as_feature_array(features: Any) -> np.ndarray:
    if isinstance(features, pd.DataFrame):
        return features.fillna(0).to_numpy()
    if isinstance(features, pd.Series):
        return features.fillna(0).to_numpy().reshape(-1, 1)
    array = np.asarray(features)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _manifest(
    dataset_id: str,
    artifacts: CandidateDiscoveryArtifacts,
    source_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    top_k: int,
    model: Any,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "generated_at": _now_iso(),
        "requires_verified_candidates": True,
        "requires_verification_sources": True,
        "candidate_library_contract_version": CANDIDATE_LIBRARY_CONTRACT_VERSION,
        "input_rows": artifacts.input_count,
        "ranked_rows": artifacts.ranked_count,
        "top_k": top_k,
        "model_class": model.__class__.__name__,
        "input_columns": list(source_df.columns),
        "ranked_columns": list(ranked_df.columns),
        "outputs": {
            "ranked_candidates_csv": artifacts.ranked_candidates_csv.name,
            "discovery_manifest_json": artifacts.discovery_manifest_json.name,
            "audit_report_md": artifacts.audit_report_md.name,
        },
    }


def _audit_report(
    dataset_id: str,
    artifacts: CandidateDiscoveryArtifacts,
    ranked_df: pd.DataFrame,
) -> str:
    lines = [
        f"# Verified Candidate Discovery Audit: {dataset_id}",
        "",
        "- Gate: verified-only candidate pool (`verification_status=verified`).",
        f"- Candidate library contract: `{CANDIDATE_LIBRARY_CONTRACT_VERSION}`.",
        f"- Input candidates: {artifacts.input_count}",
        f"- Ranked candidates emitted: {artifacts.ranked_count}",
        "",
        "## Top Candidates",
        "",
    ]
    if ranked_df.empty:
        lines.append("- None")
    else:
        for _, row in ranked_df.head(10).iterrows():
            record_id = _text(row.get("candidate_id")) or _text(row.get("record_id")) or "unknown"
            smiles = _text(row.get("smiles")) or "unknown"
            score = row.get("predicted_delta_pce")
            lines.append(f"- {record_id}: `{smiles}` predicted_delta_pce={score}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{artifacts.ranked_candidates_csv.name}`",
            f"- `{artifacts.discovery_manifest_json.name}`",
            f"- `{artifacts.audit_report_md.name}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _parse_verification_sources(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [source for source in value if isinstance(source, dict)]
    if not _text(value):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [source for source in parsed if isinstance(source, dict)]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
