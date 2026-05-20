"""guardrail.py — Input/output validation and safety checks."""

import re
from typing import List, Dict, Tuple


class Guardrail:
    """Validation utilities for SMILES strings, PCE values, metrics, and references."""

    # ------------------------------------------------------------------ #
    # SMILES
    # ------------------------------------------------------------------ #
    def validate_smiles(self, smiles: str) -> bool:
        if not smiles or not isinstance(smiles, str):
            return False
        s = smiles.strip()
        if len(s) == 0:
            return False
        # Try RDKit when available
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(s)
            return mol is not None
        except Exception:
            # Fallback: basic pattern sanity check
            pattern = re.compile(r"^[A-Za-z0-9\[\]\(\)\=\#\$\:\@\.\+\-\\\\\/\*\%\{\}@]+$")
            return bool(pattern.match(s))

    # ------------------------------------------------------------------ #
    # PCE
    # ------------------------------------------------------------------ #
    def validate_pce(self, pce: float) -> bool:
        if not isinstance(pce, (int, float)):
            return False
        # Physical bounds for perovskite solar cells (0 – 35 %)
        return 0.0 <= float(pce) <= 35.0

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def validate_metrics(self, metrics: dict) -> Tuple[bool, str]:
        if not isinstance(metrics, dict):
            return False, "metrics must be a dict"

        r2 = metrics.get("r2")
        if r2 is not None:
            if not isinstance(r2, (int, float)):
                return False, "R2 must be numeric"
            if not (-1.0 <= float(r2) <= 1.0):
                return False, f"R2 out of range [-1, 1]: {r2}"

        rmse = metrics.get("rmse")
        if rmse is not None:
            if not isinstance(rmse, (int, float)):
                return False, "RMSE must be numeric"
            if float(rmse) < 0:
                return False, f"RMSE must be >= 0: {rmse}"

        mae = metrics.get("mae")
        if mae is not None:
            if not isinstance(mae, (int, float)):
                return False, "MAE must be numeric"
            if float(mae) < 0:
                return False, f"MAE must be >= 0: {mae}"

        pearson_r = metrics.get("pearson_r")
        if pearson_r is not None:
            if not isinstance(pearson_r, (int, float)):
                return False, "pearson_r must be numeric"
            if not (-1.0 <= float(pearson_r) <= 1.0):
                return False, f"pearson_r out of range [-1, 1]: {pearson_r}"

        return True, ""

    # ------------------------------------------------------------------ #
    # References
    # ------------------------------------------------------------------ #
    def check_duplicate_references(self, refs: List[Dict]) -> List[Dict]:
        """Return *refs* with duplicates removed, keyed by ``doi``."""
        if not refs:
            return refs
        seen: set[str] = set()
        unique: List[Dict] = []
        for ref in refs:
            doi = ref.get("doi")
            if doi is not None:
                key = str(doi).strip().lower()
                if key in seen:
                    continue
                seen.add(key)
            unique.append(ref)
        return unique

    def verify_reference_integrity(self, refs: List[Dict]) -> Tuple[bool, List[str]]:
        """Ensure every ref has DOI and a minimal set of required fields."""
        if not refs:
            return True, []

        errors: List[str] = []
        required = {"doi", "title", "year"}
        for i, ref in enumerate(refs):
            missing = required - set(ref.keys())
            if missing:
                errors.append(f"Ref {i}: missing fields {sorted(missing)}")
                continue
            if not str(ref.get("doi", "")).strip():
                errors.append(f"Ref {i}: empty DOI")
            if not str(ref.get("title", "")).strip():
                errors.append(f"Ref {i}: empty title")
            year = ref.get("year")
            if year is None or (isinstance(year, (int, float)) and year < 1900):
                errors.append(f"Ref {i}: invalid year {year}")

        return len(errors) == 0, errors
