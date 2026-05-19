"""Real-data authenticity checks for PSC additive datasets.

The checker is strict by default: records must have traceable literature and
molecule evidence before they can enter the default training set. External
verification is injected through resolver callables so tests stay deterministic
and production code can use Crossref, PubChem, or MCP-backed resolvers.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from .guardrail import Guardrail


ReferenceVerifier = Callable[[str], "ReferenceEvidence | None"]
MoleculeVerifier = Callable[[dict[str, Any]], "MoleculeEvidence | None"]


@dataclass(frozen=True)
class ReferenceEvidence:
    """External or local evidence proving a publication record exists."""

    doi: str
    title: str
    year: int | None = None
    journal: str | None = None
    source: str = "reference"
    url: str | None = None

    def to_source(self) -> dict[str, Any]:
        return {
            "kind": "reference",
            "source": self.source,
            "doi": self.doi,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "url": self.url,
        }


@dataclass(frozen=True)
class MoleculeEvidence:
    """External or local evidence proving a molecule identity exists."""

    smiles: str
    pubchem_id: str | None = None
    cas_number: str | None = None
    source: str = "molecule"
    url: str | None = None

    def to_source(self) -> dict[str, Any]:
        return {
            "kind": "molecule",
            "source": self.source,
            "smiles": self.smiles,
            "pubchem_id": self.pubchem_id,
            "cas_number": self.cas_number,
            "url": self.url,
        }


@dataclass(frozen=True)
class VerificationResult:
    """Authenticity decision for one input row."""

    status: str
    reasons: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    normalized: dict[str, Any] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        return self.status == "verified"


@dataclass(frozen=True)
class VerificationSplit:
    """Strict split between model-ready rows and quarantined rows."""

    verified: list[dict[str, Any]]
    quarantine: list[dict[str, Any]]


class RealDataAuthenticator:
    """Validate literature, molecule identity, and JV-derived target values."""

    def __init__(
        self,
        reference_verifier: ReferenceVerifier | None = None,
        molecule_verifier: MoleculeVerifier | None = None,
        *,
        strict: bool = True,
        delta_tolerance: float = 0.05,
    ) -> None:
        self.reference_verifier = reference_verifier
        self.molecule_verifier = molecule_verifier
        self.strict = strict
        self.delta_tolerance = delta_tolerance
        self.guardrail = Guardrail()

    def verify_record(self, record: dict[str, Any]) -> VerificationResult:
        reasons: list[str] = []
        sources: list[dict[str, Any]] = []
        normalized = dict(record)

        doi = _clean_text(record.get("doi"))
        title = _clean_text(record.get("title"))
        smiles = _clean_text(record.get("smiles"))

        if not doi:
            reasons.append("missing_doi")
        if not title:
            reasons.append("missing_title")

        if smiles:
            if not self.guardrail.validate_smiles(smiles):
                reasons.append("invalid_smiles")
        else:
            reasons.append("missing_smiles")

        reference = self.reference_verifier(doi) if doi and self.reference_verifier else None
        if reference:
            sources.append(reference.to_source())
            if title and not _titles_match(title, reference.title):
                reasons.append("reference_title_mismatch")
            record_year = _extract_year(record)
            if record_year is not None and reference.year is not None and record_year != reference.year:
                reasons.append("reference_year_mismatch")
        elif self.strict:
            reasons.append("missing_reference_evidence")

        molecule = self.molecule_verifier(record) if self.molecule_verifier else None
        if molecule:
            sources.append(molecule.to_source())
            if smiles and molecule.smiles and not _smiles_match(smiles, molecule.smiles):
                reasons.append("molecule_smiles_mismatch")
            pubchem_id = _normalize_pubchem_id(record.get("pubchem_id"))
            molecule_pubchem_id = _normalize_pubchem_id(molecule.pubchem_id)
            if pubchem_id and molecule_pubchem_id and pubchem_id != molecule_pubchem_id:
                reasons.append("pubchem_id_mismatch")
            cas_number = _clean_text(record.get("cas_number"))
            if cas_number and molecule.cas_number and cas_number != str(molecule.cas_number):
                reasons.append("cas_number_mismatch")
        elif self.strict:
            reasons.append("missing_molecule_evidence")

        self._validate_jv_values(record, reasons, normalized)

        status = "verified" if not reasons else "quarantine"
        normalized["verification_status"] = status
        if reasons:
            normalized["quarantine_reason"] = ";".join(reasons)
        normalized["verification_sources"] = sources

        return VerificationResult(status=status, reasons=reasons, sources=sources, normalized=normalized)

    def split_records(self, records: Iterable[dict[str, Any]]) -> VerificationSplit:
        verified: list[dict[str, Any]] = []
        quarantine: list[dict[str, Any]] = []
        for record in records:
            result = self.verify_record(record)
            if result.is_verified:
                verified.append(result.normalized)
            else:
                quarantine.append(result.normalized)
        return VerificationSplit(verified=verified, quarantine=quarantine)

    def _validate_jv_values(
        self,
        record: dict[str, Any],
        reasons: list[str],
        normalized: dict[str, Any],
    ) -> None:
        baseline = _to_float(record.get("jv_reverse_scan_pce_without_modulator"))
        treated = _to_float(record.get("jv_reverse_scan_pce"))
        delta = _to_float(record.get("delta_pce"))

        if baseline is None:
            reasons.append("missing_baseline_pce")
        elif not self.guardrail.validate_pce(baseline):
            reasons.append("baseline_pce_out_of_bounds")
        else:
            normalized["jv_reverse_scan_pce_without_modulator"] = baseline

        if treated is None:
            reasons.append("missing_treated_pce")
        elif not self.guardrail.validate_pce(treated):
            reasons.append("treated_pce_out_of_bounds")
        else:
            normalized["jv_reverse_scan_pce"] = treated

        if delta is None:
            if baseline is not None and treated is not None:
                normalized["delta_pce"] = round(treated - baseline, 6)
            else:
                reasons.append("missing_delta_pce")
            return

        normalized["delta_pce"] = delta
        if baseline is not None and treated is not None:
            expected = treated - baseline
            if abs(delta - expected) > self.delta_tolerance:
                reasons.append("delta_pce_mismatch")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


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
        value = _clean_text(record.get(key))
        if not value:
            continue
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return int(match.group(0))
    return None


def _normalize_title(title: str) -> str:
    text = title.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _titles_match(left: str, right: str) -> bool:
    left_norm = _normalize_title(left)
    right_norm = _normalize_title(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    return SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.92


def _smiles_match(left: str, right: str) -> bool:
    return left.strip() == right.strip()


class CachedReferenceVerifier:
    """JSON-backed DOI evidence cache around a reference verifier."""

    def __init__(self, verifier: ReferenceVerifier, cache_path: str | Path) -> None:
        self.verifier = verifier
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict[str, Any] | None] = self._load_cache()

    def __call__(self, doi: str) -> ReferenceEvidence | None:
        key = _normalize_doi(doi)
        if not key:
            return None
        if key in self.cache:
            return _reference_from_source(self.cache[key])
        evidence = self.verifier(key)
        self.cache[key] = evidence.to_source() if evidence else None
        self._write_cache()
        return evidence

    def _load_cache(self) -> dict[str, dict[str, Any] | None]:
        if not self.cache_path.exists():
            return {}
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Reference cache must be a JSON object: {self.cache_path}")
        return payload

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class CachedMoleculeVerifier:
    """JSON-backed molecule evidence cache keyed by normalized PubChem CID."""

    def __init__(self, verifier: MoleculeVerifier, cache_path: str | Path) -> None:
        self.verifier = verifier
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict[str, Any] | None] = self._load_cache()

    def __call__(self, record: dict[str, Any]) -> MoleculeEvidence | None:
        key = _molecule_cache_key(record)
        if not key:
            return None
        if key in self.cache:
            return _molecule_from_source(self.cache[key])
        evidence = self.verifier(record)
        self.cache[key] = evidence.to_source() if evidence else None
        self._write_cache()
        return evidence

    def _load_cache(self) -> dict[str, dict[str, Any] | None]:
        if not self.cache_path.exists():
            return {}
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Molecule cache must be a JSON object: {self.cache_path}")
        return payload

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _molecule_cache_key(record: dict[str, Any]) -> str:
    pubchem_id = _normalize_pubchem_id(record.get("pubchem_id"))
    if pubchem_id:
        return f"pubchem:{pubchem_id}"
    smiles = _clean_text(record.get("smiles"))
    if smiles:
        return f"smiles:{smiles}"
    return ""


def _reference_from_source(payload: dict[str, Any] | None) -> ReferenceEvidence | None:
    if not payload:
        return None
    return ReferenceEvidence(
        doi=_clean_text(payload.get("doi")),
        title=_clean_text(payload.get("title")),
        year=_source_int(payload.get("year")),
        journal=_clean_text(payload.get("journal")) or None,
        source=_clean_text(payload.get("source")) or "reference",
        url=_clean_text(payload.get("url")) or None,
    )


def _molecule_from_source(payload: dict[str, Any] | None) -> MoleculeEvidence | None:
    if not payload:
        return None
    return MoleculeEvidence(
        smiles=_clean_text(payload.get("smiles")),
        pubchem_id=_normalize_pubchem_id(payload.get("pubchem_id")) or None,
        cas_number=_clean_text(payload.get("cas_number")) or None,
        source=_clean_text(payload.get("source")) or "molecule",
        url=_clean_text(payload.get("url")) or None,
    )


def _source_int(value: Any) -> int | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


class CrossrefReferenceVerifier:
    """Simple Crossref resolver used by production code when network is allowed."""

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def __call__(self, doi: str) -> ReferenceEvidence | None:
        doi = _normalize_doi(doi)
        if not doi:
            return None
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url, timeout=self.timeout)
        if response.status_code != 200:
            return None
        message = response.json().get("message", {})
        titles = message.get("title") or []
        container = message.get("container-title") or []
        year = _crossref_year(message)
        return ReferenceEvidence(
            doi=message.get("DOI", doi),
            title=titles[0] if titles else "",
            year=year,
            journal=container[0] if container else None,
            source="crossref",
            url=message.get("URL") or f"https://doi.org/{doi}",
        )


class PubChemMoleculeVerifier:
    """Simple PubChem resolver for CID-backed molecule evidence."""

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def __call__(self, record: dict[str, Any]) -> MoleculeEvidence | None:
        cid = _normalize_pubchem_id(record.get("pubchem_id"))
        if not cid:
            return None
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cid}/property/CanonicalSMILES/JSON"
        )
        response = requests.get(url, timeout=self.timeout)
        if response.status_code != 200:
            return None
        props = response.json().get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None
        smiles = props[0].get("CanonicalSMILES")
        if not smiles:
            return None
        return MoleculeEvidence(
            smiles=smiles,
            pubchem_id=cid,
            cas_number=_clean_text(record.get("cas_number")) or None,
            source="pubchem",
            url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        )


def _crossref_year(message: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued"):
        parts = message.get(key, {}).get("date-parts", [])
        if parts and parts[0]:
            return int(parts[0][0])
    return None
