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


def test_quality_score_downgrades_failed_review_or_audit():
    from reporting.top_journal_report import TopJournalReport

    report = TopJournalReport([], {})
    score = report._quality_score(
        8,
        {"passed": False, "findings": ["missing references"]},
        {"passed": True, "unsupported_claims": []},
    )

    assert score < 0.7
