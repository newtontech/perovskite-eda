"""Plan registry for top-journal research-story generation.

The registry is intentionally data-driven: each plan declares the evidence it
needs and the quality gates it enforces. Report generators can evaluate the
same registry into a machine-readable manifest and optionally fail fast when a
gate is not satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_REGISTRY_PATH = PROJECT_ROOT / "configs" / "plan_registry.yaml"
REQUIRED_STAGES = ("science", "validation", "figure", "writing", "review")


class PlanRegistryError(RuntimeError):
    """Raised when a plan registry is invalid or a required plan gate fails."""


@dataclass(frozen=True)
class PlanDefinition:
    """One plan entry loaded from the YAML registry."""

    id: str
    stage: str
    description: str
    inputs: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    required_evidence: tuple[str, ...] = field(default_factory=tuple)
    quality_gates: dict[str, Any] = field(default_factory=dict)
    failure_policy: str = "warn"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PlanDefinition":
        required = ("id", "stage", "description", "inputs", "outputs")
        missing = [key for key in required if key not in data]
        if missing:
            raise PlanRegistryError(f"plan is missing required field(s): {', '.join(missing)}")
        return cls(
            id=str(data["id"]),
            stage=str(data["stage"]),
            description=str(data["description"]),
            inputs=tuple(str(item) for item in data.get("inputs", [])),
            outputs=tuple(str(item) for item in data.get("outputs", [])),
            required_evidence=tuple(str(item) for item in data.get("required_evidence", [])),
            quality_gates=dict(data.get("quality_gates", {}) or {}),
            failure_policy=str(data.get("failure_policy", "warn")),
        )


@dataclass(frozen=True)
class PlanEvaluation:
    """Result of evaluating one plan against available evidence."""

    id: str
    stage: str
    status: str
    missing_evidence: tuple[str, ...] = field(default_factory=tuple)
    failed_gates: tuple[str, ...] = field(default_factory=tuple)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "status": self.status,
            "missing_evidence": list(self.missing_evidence),
            "failed_gates": list(self.failed_gates),
        }


@dataclass(frozen=True)
class PlanRegistry:
    """Validated collection of stage-ordered plan definitions."""

    plans: tuple[PlanDefinition, ...]
    version: int = 1
    artifact_policy: str = "evidence-light"
    path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "PlanRegistry":
        registry_path = Path(path)
        with open(registry_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        registry = cls.from_mapping(data)
        return cls(
            plans=registry.plans,
            version=registry.version,
            artifact_policy=registry.artifact_policy,
            path=registry_path,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PlanRegistry":
        plans = tuple(PlanDefinition.from_mapping(item) for item in data.get("plans", []))
        registry = cls(
            plans=plans,
            version=int(data.get("version", 1)),
            artifact_policy=str(data.get("artifact_policy", "evidence-light")),
        )
        registry.validate()
        return registry

    def validate(self) -> None:
        if not self.plans:
            raise PlanRegistryError("plan registry has no plans")
        stages = tuple(plan.stage for plan in self.plans)
        if stages != REQUIRED_STAGES:
            raise PlanRegistryError(
                f"plan registry stages must be {list(REQUIRED_STAGES)}, got {list(stages)}"
            )
        ids = [plan.id for plan in self.plans]
        duplicates = sorted({plan_id for plan_id in ids if ids.count(plan_id) > 1})
        if duplicates:
            raise PlanRegistryError(f"duplicate plan id(s): {', '.join(duplicates)}")

    def evaluate(self, context: dict[str, Any]) -> list[PlanEvaluation]:
        return [self._evaluate_plan(plan, context) for plan in self.plans]

    def require_passed(self, context: dict[str, Any]) -> list[PlanEvaluation]:
        evaluations = self.evaluate(context)
        failures = [item for item in evaluations if item.status != "passed"]
        if failures:
            details = []
            for item in failures:
                pieces = []
                if item.missing_evidence:
                    pieces.append(f"missing evidence: {', '.join(item.missing_evidence)}")
                if item.failed_gates:
                    pieces.append(f"failed gates: {', '.join(item.failed_gates)}")
                details.append(f"{item.id} ({'; '.join(pieces)})")
            raise PlanRegistryError("Plan registry gate failed: " + " | ".join(details))
        return evaluations

    def to_manifest(self, evaluations: list[PlanEvaluation]) -> dict[str, Any]:
        return {
            "version": self.version,
            "path": str(self.path) if self.path else None,
            "artifact_policy": self.artifact_policy,
            "plans": [item.to_manifest() for item in evaluations],
        }

    def _evaluate_plan(self, plan: PlanDefinition, context: dict[str, Any]) -> PlanEvaluation:
        missing = tuple(
            evidence for evidence in plan.required_evidence
            if not _evidence_present(context.get(evidence))
        )
        failed = tuple(_failed_quality_gates(plan.quality_gates, context))
        status = "passed" if not missing and not failed else "failed"
        return PlanEvaluation(
            id=plan.id,
            stage=plan.stage,
            status=status,
            missing_evidence=missing,
            failed_gates=failed,
        )


def load_plan_registry(path: str | Path | None = None) -> PlanRegistry:
    """Load the default or user-supplied plan registry."""

    return PlanRegistry.from_file(path or DEFAULT_PLAN_REGISTRY_PATH)


def build_report_plan_context(
    *,
    results: list[dict],
    artifacts: dict[str, Any],
    figures: list[Path],
    report_text: str,
    claim_ledger: list[dict],
    review: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Build evidence booleans and counts used by the plan registry."""

    successful = [row for row in results if row.get("status") == "success"]
    splits = [
        row.get("config", {}).get("layer4", {}).get("method_id", "")
        for row in successful
    ]
    metric_claims = [
        item for item in claim_ledger
        if str(item.get("evidence_id", "")).startswith("metric:")
    ]
    figure_claims = [
        item for item in claim_ledger
        if str(item.get("evidence_id", "")).startswith("figure:")
    ]
    references = review.get("reference_count", 0)
    return {
        "successful_runs": len(successful),
        "successful_results": bool(successful),
        "scaffold_split": any("scaffold" in split.lower() for split in splits),
        "prediction_arrays": bool(artifacts.get("y_true") and artifacts.get("y_pred")),
        "shap_diagnostics": bool(artifacts.get("shap_values")),
        "external_validation": bool(artifacts.get("y_true_external") and artifacts.get("y_pred_external")),
        "main_figures": len(figures),
        "figures_in_context": "## Figures" not in report_text and "**Figure " in report_text,
        "claim_ledger": bool(claim_ledger),
        "metric_claims": len(metric_claims),
        "figure_claims": len(figure_claims),
        "references": references,
        "review_report": bool(review),
        "review_passed": bool(review.get("passed")),
        "audit_passed": bool(audit.get("passed")),
    }


def _evidence_present(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value > 0
    return bool(value)


def _failed_quality_gates(gates: dict[str, Any], context: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for gate, expected in gates.items():
        actual = context.get(_context_key_for_gate(gate), context.get(gate))
        if gate.startswith("min_"):
            if float(actual or 0) < float(expected):
                failed.append(f"{gate}>={expected}")
        elif gate.startswith("max_"):
            if actual is not None and float(actual) > float(expected):
                failed.append(f"{gate}<={expected}")
        elif gate.startswith("require_"):
            if bool(expected) and not bool(actual):
                failed.append(gate)
        elif actual != expected:
            failed.append(f"{gate}={expected}")
    return failed


def _context_key_for_gate(gate: str) -> str:
    aliases = {
        "min_successful_runs": "successful_runs",
        "min_main_figures": "main_figures",
        "max_main_figures": "main_figures",
        "min_metric_claims": "metric_claims",
        "min_figure_claims": "figure_claims",
        "min_references": "references",
        "require_scaffold_split": "scaffold_split",
        "require_prediction_arrays": "prediction_arrays",
        "require_shap_diagnostics": "shap_diagnostics",
        "require_figures_in_context": "figures_in_context",
        "require_claim_ledger": "claim_ledger",
        "require_review_passed": "review_passed",
        "require_audit_passed": "audit_passed",
        "require_external_validation": "external_validation",
    }
    return aliases.get(gate, gate)
