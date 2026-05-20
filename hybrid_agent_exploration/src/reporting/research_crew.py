"""Lightweight stateful research-crew agents for report quality control."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OVERCLAIM_PATTERNS = (
    "strong correlation",
    "strong linear agreement",
    "highly accurate",
    "excellent performance",
    "state-of-the-art",
    "external validation was performed",
)

CANDIDATE_COUNT_PATTERN = re.compile(
    r"\b(?P<count>\d{1,3}(?:,\d{3})+|\d+)\s+(?P<source>PubChem)\s+candidates?\b",
    flags=re.IGNORECASE,
)
SAM_PCE_PATTERN = re.compile(
    r"\b(?P<candidate>SAM-\d+)\b(?:(?!\n\n).){0,120}?"
    r"(?P<pce>\d{1,2}(?:\.\d+)?)\s*%",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass
class PlannerAgent:
    """Describe the intended top-journal report product."""

    quality_target: str = "top-journal"

    def plan(self) -> dict[str, Any]:
        return {
            "quality_target": self.quality_target,
            "required_sections": [
                "Abstract",
                "Introduction",
                "Methods",
                "Results and Discussion",
                "Limitations",
                "Conclusion",
            ],
            "minimum_main_figures": 8 if self.quality_target == "top-journal" else 5,
            "evidence_policy": "Every quantitative or mechanistic claim must map to a metric, figure, or declared evidence gap.",
        }


@dataclass
class ReviewerAgent:
    """Review generated text against manuscript-format and artifact gates."""

    minimum_figures: int = 8

    def review(self, text: str, figures: list[Path], claim_ledger: list[dict]) -> dict[str, Any]:
        findings: list[str] = []
        required_sections = [
            "## Abstract",
            "## 1. Introduction",
            "## 2. Methods",
            "## 3. Results and Discussion",
            "## 4. Limitations",
            "## Conclusion",
        ]
        for section in required_sections:
            if section not in text:
                findings.append(f"missing section: {section}")
        if "## Figures" in text:
            findings.append("figures are still collected in a terminal section")
        if len(figures) < self.minimum_figures:
            findings.append(f"only {len(figures)} main figures, expected at least {self.minimum_figures}")
        if not claim_ledger:
            findings.append("claim ledger is empty")
        if not any(item.get("evidence_id", "").startswith("metric:") for item in claim_ledger):
            findings.append("no metric evidence entries in claim ledger")
        if not any(item.get("evidence_id", "").startswith("figure:") for item in claim_ledger):
            findings.append("no figure evidence entries in claim ledger")
        reference_count = 0
        if "## References" in text:
            reference_text = text.split("## References", 1)[1]
            reference_count = len(re.findall(r"^\d+\. ", reference_text, flags=re.MULTILINE))
            if reference_count < 30:
                findings.append(f"only {reference_count} references, expected at least 30 for top-journal target")
        else:
            findings.append("missing references section")

        return {
            "passed": not findings,
            "findings": findings,
            "figure_count": len(figures),
            "claim_count": len(claim_ledger),
            "reference_count": reference_count,
            "rubric": {
                "figures_in_context": "## Figures" not in text and len(figures) >= self.minimum_figures,
                "claim_evidence_audit": bool(claim_ledger),
                "literature_benchmark": reference_count >= 30,
                "scaffold_generalization_disclosed": "scaffold" in text.lower(),
                "ml_innovation": all(token in text.lower() for token in ["claim ledger", "scaffold", "shap"]),
            },
        }


@dataclass
class ClaimAuditorAgent:
    """Audit report claims against the strongest supported R2 value."""

    max_supported_r2: float | None = None

    def sanitize_text(self, text: str) -> str:
        if self.max_supported_r2 is None or self.max_supported_r2 >= 0.3:
            return text
        replacements = {
            "strong correlation": "modest predictive signal",
            "strong linear agreement": "modest linear agreement",
            "highly accurate": "exploratory",
            "excellent performance": "limited performance",
            "state-of-the-art": "run-derived",
        }
        sanitized = text
        for phrase, replacement in replacements.items():
            sanitized = re.sub(re.escape(phrase), replacement, sanitized, flags=re.IGNORECASE)
        return sanitized

    def audit_text(self, text: str, claim_ledger: list[dict]) -> dict[str, Any]:
        unsupported: list[dict[str, str]] = []
        lowered = text.lower()
        ledger_text = " ".join(str(item.get("evidence_id", "")) for item in claim_ledger)
        for phrase in OVERCLAIM_PATTERNS:
            if phrase in lowered:
                if phrase == "external validation was performed" and "metric:external_validation" in ledger_text:
                    continue
                if self.max_supported_r2 is None or self.max_supported_r2 < 0.3:
                    unsupported.append(
                        {
                            "phrase": phrase,
                            "reason": "phrase exceeds the evidence supported by current metrics",
                        }
                    )
        unsupported.extend(self._audit_candidate_discovery_claims(text, claim_ledger))
        missing_evidence = [
            item for item in claim_ledger
            if not item.get("evidence_id")
        ]
        return {
            "passed": not unsupported and not missing_evidence,
            "unsupported_claims": unsupported,
            "missing_evidence": missing_evidence,
        }

    def _audit_candidate_discovery_claims(self, text: str, claim_ledger: list[dict]) -> list[dict[str, str]]:
        unsupported: list[dict[str, str]] = []
        for match in CANDIDATE_COUNT_PATTERN.finditer(text):
            raw_count = match.group("count")
            count = int(raw_count.replace(",", ""))
            source = match.group("source")
            if not self._has_candidate_count_evidence(claim_ledger, count, source):
                unsupported.append(
                    {
                        "phrase": f"{raw_count} {source} candidates",
                        "reason": "candidate-pool size claim lacks manifest-backed evidence",
                    }
                )

        for match in SAM_PCE_PATTERN.finditer(text):
            candidate = match.group("candidate").upper()
            pce = float(match.group("pce"))
            if not self._has_candidate_pce_evidence(claim_ledger, candidate, pce):
                unsupported.append(
                    {
                        "phrase": f"{candidate} predicted PCE {pce:g}%",
                        "reason": "candidate predicted-performance claim lacks manifest-backed evidence",
                    }
                )

        return unsupported

    @staticmethod
    def _has_candidate_count_evidence(claim_ledger: list[dict], count: int, source: str) -> bool:
        source_lower = source.lower()
        for item in claim_ledger:
            evidence_id = str(item.get("evidence_id", "")).lower()
            if not any(token in evidence_id for token in ("manifest", "discovery", "candidate")):
                continue
            haystack = " ".join(
                str(item.get(key, ""))
                for key in ("claim", "source", "dataset", "candidate_source")
            ).lower()
            if source_lower not in haystack and source_lower not in evidence_id:
                continue
            if any(
                ClaimAuditorAgent._numeric_matches(item.get(key), count)
                for key in ("value", "count", "candidate_count", "n_candidates", "ranked_candidates")
            ):
                return True
        return False

    @staticmethod
    def _has_candidate_pce_evidence(claim_ledger: list[dict], candidate: str, pce: float) -> bool:
        candidate_lower = candidate.lower()
        for item in claim_ledger:
            evidence_id = str(item.get("evidence_id", "")).lower()
            haystack = " ".join(
                str(item.get(key, ""))
                for key in ("claim", "candidate", "candidate_id", "name")
            ).lower()
            if candidate_lower not in haystack and candidate_lower not in evidence_id:
                continue
            if any(
                ClaimAuditorAgent._numeric_matches(item.get(key), pce)
                for key in ("value", "pce", "predicted_pce", "predicted_pce_percent")
            ):
                return True
        return False

    @staticmethod
    def _numeric_matches(value: Any, expected: float | int) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            return any(ClaimAuditorAgent._numeric_matches(item, expected) for item in value.values())
        try:
            observed = float(str(value).replace(",", "").rstrip("%"))
        except (TypeError, ValueError):
            return False
        return abs(observed - float(expected)) < 1e-6


@dataclass
class EvidenceAgent:
    """Summarize evidence classes available to the report."""

    def summarize(self, results: list[dict], artifacts: dict[str, Any]) -> dict[str, Any]:
        successful = [row for row in results if row.get("status") == "success"]
        splits = [
            row.get("config", {}).get("layer4", {}).get("method_id", "")
            for row in successful
        ]
        return {
            "successful_runs": len(successful),
            "scaffold_split_runs": sum("scaffold" in split.lower() for split in splits),
            "has_prediction_arrays": bool(artifacts.get("y_true") and artifacts.get("y_pred")),
            "has_shap": bool(artifacts.get("shap_values")),
            "has_external_validation": bool(artifacts.get("y_true_external") and artifacts.get("y_pred_external")),
        }
