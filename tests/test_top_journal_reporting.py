import base64
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _sample_results() -> list[dict]:
    rows = []
    agent = 1
    for feature in ("F21_rdkit_basic", "F22_morgan_256"):
        for model in ("M31_random_forest", "M31_xgboost"):
            for split, r2 in (
                ("E42_random_split", 0.24),
                ("E42_scaffold_split", -0.06),
            ):
                rows.append(
                    {
                        "agent_id": f"agent_{agent:03d}",
                        "status": "success",
                        "n_samples": 128,
                        "n_features": 16 if feature == "F21_rdkit_basic" else 257,
                        "duration_sec": 1.2,
                        "config": {
                            "layer1": {"method_id": "agentic_veryloose"},
                            "layer2": {"method_id": feature},
                            "layer3": {"method_id": model},
                            "layer4": {"method_id": split},
                            "layer5": {"method_id": "D54_report_only"},
                            "target": "delta_pce",
                            "baseline_as_feature": True,
                            "_hash": f"{feature}_{model}_{split}",
                        },
                        "metrics": {
                            "r2": r2,
                            "rmse": 2.4 if r2 > 0 else 3.7,
                            "mae": 1.8,
                            "pearson_r": 0.42 if r2 > 0 else 0.12,
                        },
                    }
                )
                agent += 1
    return rows


def _sample_artifacts() -> dict:
    y_true = [0.0, 1.0, 2.0, 3.0]
    y_pred = [0.1, 0.9, 1.8, 2.7]
    multi = []
    for idx, result in enumerate(_sample_results()):
        cfg = result["config"]
        multi.append(
            {
                "name": f"{cfg['layer3']['method_id']} + {cfg['layer2']['method_id']}",
                "agent_id": result["agent_id"],
                "r2": result["metrics"]["r2"],
                "rmse": result["metrics"]["rmse"],
                "mae": result["metrics"]["mae"],
                "pearson_r": result["metrics"]["pearson_r"],
                "y_true": y_true,
                "y_pred": [v + idx * 0.01 for v in y_pred],
                "feature_importances": [0.4, 0.3, 0.2, 0.1],
            }
        )
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "feature_importances": [0.4, 0.3, 0.2, 0.1],
        "shap_values": [[0.1, -0.2, 0.05, 0.0], [0.2, -0.1, 0.04, 0.0]],
        "shap_background": [[1, 0, 0, 1], [0, 1, 0, 1]],
        "multi_model_results": multi,
    }


def _write_verified_discovery_artifact_dir(root: Path) -> Path:
    artifact_dir = root / "verified_discovery"
    (artifact_dir / "discovery").mkdir(parents=True)
    (artifact_dir / "dataset").mkdir(parents=True)
    (artifact_dir / "workflow_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "verified-fixture",
                "artifact_policy": "verified-light-artifacts-in-git",
                "verified_rows": 12,
                "quarantine_rows": 3,
                "ranked_candidates": 3,
                "top_k": 3,
                "outputs": {
                    "ranked_candidates_csv": "discovery/ranked_candidates.csv",
                    "doi_manifest_json": "dataset/doi_manifest.json",
                    "quarantine_csv": "dataset/quarantine.csv",
                },
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "discovery" / "ranked_candidates.csv").write_text(
        "\n".join(
            [
                "rank,record_id,smiles,predicted_delta_pce,uncertainty,doi,verification_status",
                "1,row-004,CCCC,1.25,0.10,10.1021/acs.jpclett.6c00122,verified",
                "2,row-003,CCC,0.75,0.20,10.1021/acs.jpclett.6c00121,verified",
                "3,row-002,CC,0.50,0.30,10.1021/acs.jpclett.6c00120,verified",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "dataset" / "doi_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "verified-fixture",
                "reference_count": 2,
                "references": [
                    {"doi": "10.1021/acs.jpclett.6c00122", "title": "Verified source A"},
                    {"doi": "10.1021/acs.jpclett.6c00121", "title": "Verified source B"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "dataset" / "quarantine.csv").write_text(
        "\n".join(
            [
                "record_id,quarantine_reason",
                "row-009,missing_doi",
                "row-010,missing_doi;invalid_smiles",
                "row-011,title_conflict",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_dir


def test_top_journal_bundle_places_figures_in_context_and_records_claims(tmp_path):
    from reporting.report_bundle import ReportBundle
    from reporting.top_journal_report import TopJournalReport

    bundle = TopJournalReport(
        _sample_results(),
        _sample_artifacts(),
        output_dir=tmp_path,
        quality_target="top-journal",
    ).generate()

    assert isinstance(bundle, ReportBundle)
    text = bundle.path.read_text(encoding="utf-8")
    assert "## Figures" not in text
    assert len(bundle.figures) >= 8
    assert text.count("**Figure ") >= 8
    assert re.search(r"Figure 1[\s\S]{0,800}!\[", text)
    assert "scaffold" in text.lower()
    assert "strong correlation" not in text.lower()
    assert "highly accurate" not in text.lower()

    assert bundle.claim_ledger
    assert all(item["evidence_id"] for item in bundle.claim_ledger)
    evidence_ids = {item["evidence_id"] for item in bundle.claim_ledger}
    assert "metric:best_model.r2" in evidence_ids
    assert any(eid.startswith("figure:fig") for eid in evidence_ids)
    assert bundle.quality_score >= 0.7

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["quality_target"] == "top-journal"
    assert manifest["agents"][-2:] == ["ReviewerAgent", "ClaimAuditorAgent"]
    review = json.loads((tmp_path / "review_report.json").read_text(encoding="utf-8"))
    assert review["review"]["passed"] is True
    assert review["claim_audit"]["passed"] is True


def test_report_bundle_ingests_verified_discovery_provenance_without_manuscript_dependency(tmp_path):
    from reporting.si_generator import SIGenerator
    from reporting.top_journal_report import TopJournalReport

    artifact_dir = _write_verified_discovery_artifact_dir(tmp_path)
    artifacts = {
        **_sample_artifacts(),
        "verified_discovery_artifact_dir": artifact_dir,
        "verified_discovery_top_n": 2,
    }

    bundle = TopJournalReport(
        _sample_results(),
        artifacts,
        output_dir=tmp_path / "report",
        quality_target="top-journal",
    ).generate()
    si_path = SIGenerator(
        _sample_results(),
        artifacts,
        output_dir=tmp_path / "si",
    ).generate()

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    discovery = manifest["verified_discovery"]
    assert discovery["dataset_id"] == "verified-fixture"
    assert discovery["source_dir"] == str(artifact_dir)
    assert [row["record_id"] for row in discovery["top_candidates"]] == ["row-004", "row-003"]
    assert "row-002" not in json.dumps(discovery)
    assert discovery["quarantine_reason_summary"] == {
        "invalid_smiles": 1,
        "missing_doi": 2,
        "title_conflict": 1,
    }

    evidence_ids = {entry["evidence_id"] for entry in bundle.claim_ledger}
    assert "discovery:top_candidates.count" in evidence_ids
    assert "provenance:quarantine_reason_summary" in evidence_ids

    si_text = si_path.read_text(encoding="utf-8")
    assert "## S9. Verified Discovery Provenance" in si_text
    assert "| 1 | row-004 | `CCCC` | 1.250 | 0.100 | 10.1021/acs.jpclett.6c00122 |" in si_text
    assert "row-002" not in si_text
    assert "- missing_doi: 2" in si_text


def test_embed_markdown_images_replaces_local_figure_links(tmp_path):
    from reporting.image_embedder import embed_markdown_images

    figures = tmp_path / "figures"
    figures.mkdir()
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    (figures / "tiny.png").write_bytes(png_bytes)
    report = tmp_path / "report.md"
    report.write_text("**Figure 1**. Tiny.\n\n![tiny](figures/tiny.png)\n", encoding="utf-8")

    output = embed_markdown_images(report)
    text = output.read_text(encoding="utf-8")
    assert "figures/tiny.png" not in text
    assert "data:image/png;base64," in text
    assert "**Figure 1**. Tiny." in text


def test_embed_markdown_images_preserves_title_suffix(tmp_path):
    from reporting.image_embedder import embed_markdown_images

    figures = tmp_path / "figures"
    figures.mkdir()
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    (figures / "tiny.png").write_bytes(png_bytes)
    report = tmp_path / "report.md"
    report.write_text('![tiny](figures/tiny.png "caption")\n', encoding="utf-8")

    embed_markdown_images(report)
    text = report.read_text(encoding="utf-8")
    assert "figures/tiny.png" not in text
    assert ' "caption")' in text
    assert "data:image/png;base64," in text


def test_claim_auditor_flags_unsupported_performance_claims():
    from reporting.research_crew import ClaimAuditorAgent

    text = "The model is highly accurate and shows strong correlation."
    ledger = [{"claim": "R2 is modest", "evidence_id": "metric:best_model.r2"}]
    findings = ClaimAuditorAgent(max_supported_r2=0.24).audit_text(text, ledger)

    assert findings["unsupported_claims"]
    assert findings["passed"] is False


def test_claim_auditor_flags_candidate_discovery_claims_without_manifest_evidence():
    from reporting.research_crew import ClaimAuditorAgent

    text = (
        "We screened 5,000 PubChem candidates and discovered SAM-1 "
        "with a predicted PCE of 26.8%."
    )
    ledger = [{"claim": "R2 is modest", "evidence_id": "metric:best_model.r2"}]
    findings = ClaimAuditorAgent(max_supported_r2=0.24).audit_text(text, ledger)

    phrases = {item["phrase"] for item in findings["unsupported_claims"]}
    assert "5,000 PubChem candidates" in phrases
    assert "SAM-1 predicted PCE 26.8%" in phrases
    assert findings["passed"] is False


def test_claim_auditor_allows_candidate_discovery_claims_with_manifest_evidence():
    from reporting.research_crew import ClaimAuditorAgent

    text = (
        "We screened 5,000 PubChem candidates and discovered SAM-1 "
        "with a predicted PCE of 26.8%."
    )
    ledger = [
        {
            "claim": "Verified PubChem candidate pool size",
            "evidence_id": "manifest:verified_discovery.pubchem_candidate_count",
            "value": 5000,
            "source": "PubChem verified discovery manifest",
        },
        {
            "claim": "SAM-1 predicted PCE from verified discovery manifest",
            "evidence_id": "manifest:verified_discovery.top_candidate.sam-1",
            "candidate": "SAM-1",
            "predicted_pce_percent": 26.8,
        },
    ]
    findings = ClaimAuditorAgent(max_supported_r2=0.24).audit_text(text, ledger)

    assert findings["unsupported_claims"] == []
    assert findings["passed"] is True


def test_quality_score_downgrades_failed_review_or_audit():
    from reporting.top_journal_report import TopJournalReport

    report = TopJournalReport([], {})
    score = report._quality_score(
        8,
        {"passed": False, "findings": ["missing references"]},
        {"passed": True, "unsupported_claims": []},
    )

    assert score < 0.7


def test_default_plan_registry_covers_top_journal_workflow():
    from reporting.plan_registry import load_plan_registry

    registry = load_plan_registry()

    assert registry.artifact_policy == "evidence-light"
    assert [plan.stage for plan in registry.plans] == [
        "science",
        "validation",
        "figure",
        "writing",
        "review",
    ]
    assert {plan.id for plan in registry.plans} == {
        "science_story_plan",
        "generalization_validation_plan",
        "claim_first_figure_plan",
        "journal_writing_plan",
        "reviewer_gate_plan",
    }


def test_plan_registry_manifest_is_written_with_report_bundle(tmp_path):
    from reporting.plan_registry import load_plan_registry
    from reporting.top_journal_report import TopJournalReport

    registry = load_plan_registry()
    bundle = TopJournalReport(
        _sample_results(),
        _sample_artifacts(),
        output_dir=tmp_path,
        quality_target="top-journal",
        plan_registry=registry,
    ).generate()

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    plan_manifest = manifest["plan_registry"]
    assert plan_manifest["artifact_policy"] == "evidence-light"
    assert [entry["stage"] for entry in plan_manifest["plans"]] == [
        "science",
        "validation",
        "figure",
        "writing",
        "review",
    ]
    assert all(entry["status"] == "passed" for entry in plan_manifest["plans"])


def test_plan_registry_strict_gate_fails_missing_scaffold_and_shap():
    from reporting.plan_registry import PlanRegistryError, load_plan_registry

    registry = load_plan_registry()
    context = {
        "successful_results": True,
        "successful_runs": 4,
        "prediction_arrays": True,
        "main_figures": 5,
        "figure_claims": 5,
        "figures_in_context": True,
        "claim_ledger": True,
        "metric_claims": 3,
        "references": 40,
        "review_report": True,
        "review_passed": True,
        "audit_passed": True,
    }
    try:
        registry.require_passed(context)
    except PlanRegistryError as exc:
        message = str(exc)
    else:
        message = ""

    assert "generalization_validation_plan" in message
    assert "scaffold_split" in message
    assert "shap_diagnostics" in message
