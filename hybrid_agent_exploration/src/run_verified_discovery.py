"""CLI for strict verified PSC additive discovery runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from harness.authenticity import (
    CachedMoleculeVerifier,
    CachedReferenceVerifier,
    CrossrefReferenceVerifier,
    MoleculeEvidence,
    PubChemMoleculeVerifier,
    RealDataAuthenticator,
    ReferenceEvidence,
)
from screening.verified_discovery_workflow import VerifiedDiscoveryWorkflow


EVIDENCE_MODES = ("external-cached", "source-columns")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SourceColumnReferenceVerifier:
    """Reference verifier backed by DOI/title/year/journal columns in the input table."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.references: dict[str, ReferenceEvidence] = {}
        if "doi" not in df.columns:
            return
        for record in df.to_dict(orient="records"):
            doi = _normalize_doi(record.get("doi"))
            if not doi or doi in self.references:
                continue
            title = _clean_text(record.get("title"))
            if not title:
                continue
            self.references[doi] = ReferenceEvidence(
                doi=doi,
                title=title,
                year=_extract_year(record),
                journal=_clean_text(record.get("journal")) or None,
                source="source-columns",
                url=f"https://doi.org/{doi}",
            )

    def __call__(self, doi: str) -> ReferenceEvidence | None:
        return self.references.get(_normalize_doi(doi))


class SourceColumnMoleculeVerifier:
    """Molecule verifier backed by SMILES/PubChem/CAS columns in the input table."""

    def __call__(self, record: dict[str, Any]) -> MoleculeEvidence | None:
        smiles = _clean_text(record.get("smiles"))
        if not smiles:
            return None
        pubchem_id = _normalize_pubchem_id(record.get("pubchem_id")) or None
        return MoleculeEvidence(
            smiles=smiles,
            pubchem_id=pubchem_id,
            cas_number=_clean_text(record.get("cas_number")) or None,
            source="source-columns",
            url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem_id}" if pubchem_id else None,
        )


def build_authenticator(
    evidence_mode: str,
    df: pd.DataFrame,
    cache_dir: str | Path,
) -> RealDataAuthenticator:
    """Build the strict authenticator requested by the CLI."""

    if evidence_mode == "external-cached":
        cache_dir = Path(cache_dir)
        return RealDataAuthenticator(
            reference_verifier=CachedReferenceVerifier(
                CrossrefReferenceVerifier(),
                cache_dir / "reference_cache.json",
            ),
            molecule_verifier=CachedMoleculeVerifier(
                PubChemMoleculeVerifier(),
                cache_dir / "molecule_cache.json",
            ),
        )
    if evidence_mode == "source-columns":
        return RealDataAuthenticator(
            reference_verifier=SourceColumnReferenceVerifier(df),
            molecule_verifier=SourceColumnMoleculeVerifier(),
        )
    raise ValueError(f"Unknown evidence mode: {evidence_mode}")


def default_cache_dir(dataset_id: str) -> Path:
    """Return the ignored default evidence-cache directory for a run."""

    safe_dataset_id = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in dataset_id)
    return PROJECT_ROOT / ".cache" / "verified_discovery" / safe_dataset_id / "evidence_cache"


def load_input(path: str | Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Load a CSV/XLSX input table for verified discovery."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input table not found: {input_path}")
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path, nrows=max_rows, low_memory=False)
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(input_path, nrows=max_rows)
    raise ValueError(f"Unsupported input table format: {input_path.suffix}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/XLSX table with PSC additive rows.")
    parser.add_argument("--output-dir", required=True, help="Directory for all workflow artifacts.")
    parser.add_argument("--dataset-id", required=True, help="Stable dataset/run identifier.")
    parser.add_argument(
        "--evidence-mode",
        choices=EVIDENCE_MODES,
        default="external-cached",
        help="Evidence source. external-cached uses Crossref/PubChem with JSON caches; source-columns is a fast local smoke mode.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Evidence cache directory; defaults to hybrid_agent_exploration/.cache/verified_discovery/<dataset-id>/evidence_cache.",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Number of ranked candidates to emit.")
    parser.add_argument("--min-verified-rows", type=int, default=10, help="Minimum strict verified training rows.")
    parser.add_argument("--max-rows", type=int, help="Optional input row cap for smoke runs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    df = load_input(args.input, max_rows=args.max_rows)
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir(args.dataset_id)
    authenticator = build_authenticator(args.evidence_mode, df, cache_dir)
    run_metadata = {
        "evidence_mode": args.evidence_mode,
        "verification_level": "external_cached" if args.evidence_mode == "external-cached" else "source_columns_only",
        "publication_grade": args.evidence_mode == "external-cached",
        "source_columns_is_smoke_only": args.evidence_mode == "source-columns",
        "input_path": str(Path(args.input)),
        "max_rows": args.max_rows,
    }
    if args.evidence_mode == "external-cached":
        run_metadata["cache_dir"] = str(cache_dir)
    artifacts = VerifiedDiscoveryWorkflow(
        output_dir=output_dir,
        authenticator=authenticator,
    ).run_from_dataframe(
        df,
        dataset_id=args.dataset_id,
        top_k=args.top_k,
        min_verified_rows=args.min_verified_rows,
        run_metadata=run_metadata,
    )
    print(f"[verified-discovery] workflow_manifest={artifacts.workflow_manifest_json}")
    print(f"[verified-discovery] verified_rows={artifacts.verified_rows}")
    print(f"[verified-discovery] quarantine_rows={artifacts.quarantine_rows}")
    print(f"[verified-discovery] ranked_candidates={artifacts.ranked_candidates}")
    return 0


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _normalize_doi(value: Any) -> str:
    return _clean_text(value).lower()


def _normalize_pubchem_id(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        parsed = float(text)
    except ValueError:
        return text
    if math.isnan(parsed):
        return ""
    if parsed.is_integer():
        return str(int(parsed))
    return text


def _extract_year(record: dict[str, Any]) -> int | None:
    for key in ("year", "publication_year", "publication_date"):
        text = _clean_text(record.get(key))
        if not text:
            continue
        try:
            return int(float(text))
        except ValueError:
            for token in text.replace("-", " ").replace("/", " ").split():
                if len(token) == 4 and token.isdigit() and token.startswith(("19", "20")):
                    return int(token)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
