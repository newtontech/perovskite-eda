import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "hybrid_agent_exploration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _ranked_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "candidate_id": "candidate-001",
                "smiles": "CCCC",
                "candidate_score": 0.91,
                "predicted_delta_pce": 1.20,
                "verification_status": "verified",
                "verification_sources": '[{"kind": "molecule", "source": "pubchem"}]',
            },
            {
                "rank": 2,
                "candidate_id": "candidate-002",
                "smiles": "CCO",
                "candidate_score": 0.72,
                "predicted_delta_pce": 0.80,
                "verification_status": "verified",
                "verification_sources": '[{"kind": "molecule", "source": "pubchem"}]',
            },
        ]
    )


def _iteration_dir(tmp_path: Path) -> Path:
    from closed_loop.iteration_packet import generate_iteration_packet

    ranked_csv = tmp_path / "ranked_candidates.csv"
    _ranked_candidates().to_csv(ranked_csv, index=False)
    iteration_dir = tmp_path / "closed_loop" / "iteration_000"
    generate_iteration_packet(
        ranked_candidates_csv=ranked_csv,
        output_dir=iteration_dir,
        dataset_id="experimental-feedback-fixture",
        iteration_index=0,
        top_k=2,
    )
    return iteration_dir


def _feedback_rows(**overrides: object) -> pd.DataFrame:
    row = {
        "iteration_id": "iteration_000",
        "candidate_id": "candidate-001",
        "smiles": "CCCC",
        "experimental_status": "measured",
        "measured_pce_without_modulator": 18.0,
        "measured_pce_with_modulator": 19.1,
        "measured_delta_pce": 1.1,
        "measurement_date": "2026-05-20",
        "protocol_reference": "lab-protocol-v1",
        "lab_or_operator": "operator-a",
        "instrument_or_batch_id": "batch-001",
        "raw_data_reference": "raw/run-001.csv",
        "data_source": "unit-test-fixture",
        "notes": "synthetic parser fixture; not real experimental evidence",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _write_feedback(path: Path, rows: pd.DataFrame) -> Path:
    rows.to_csv(path, index=False)
    return path


def test_ingests_valid_experimental_feedback_without_touching_template(tmp_path):
    from closed_loop.feedback_ingestion import ingest_experimental_feedback

    iteration_dir = _iteration_dir(tmp_path)
    feedback_csv = _write_feedback(tmp_path / "feedback.csv", _feedback_rows())

    artifacts = ingest_experimental_feedback(
        iteration_dir=iteration_dir,
        feedback_csv=feedback_csv,
        feedback_data_scope="unit_test_fixture",
    )

    assert artifacts.output_dir == iteration_dir / "experimental_feedback"
    assert artifacts.validated_feedback_csv.exists()
    assert artifacts.feedback_vs_prediction_csv.exists()
    assert artifacts.feedback_ingestion_manifest_json.exists()
    assert artifacts.feedback_validation_report_json.exists()

    template = pd.read_csv(iteration_dir / "feedback_template.csv", keep_default_na=False)
    assert set(template["experimental_status"]) == {"pending_external_measurement"}
    assert template["measured_delta_pce"].tolist() == ["", ""]

    validated = pd.read_csv(artifacts.validated_feedback_csv, keep_default_na=False)
    assert validated["candidate_id"].tolist() == ["candidate-001"]
    assert validated["feedback_data_scope"].tolist() == ["unit_test_fixture"]

    comparison = pd.read_csv(artifacts.feedback_vs_prediction_csv)
    assert comparison["candidate_id"].tolist() == ["candidate-001"]
    assert comparison.loc[0, "predicted_delta_pce"] == pytest.approx(1.2)
    assert comparison.loc[0, "measured_delta_pce"] == pytest.approx(1.1)
    assert comparison.loc[0, "prediction_error_delta_pce"] == pytest.approx(0.1)

    manifest = json.loads(artifacts.feedback_ingestion_manifest_json.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "closed-loop-feedback-ingestion-manifest-v1"
    assert manifest["feedback_data_scope"] == "unit_test_fixture"
    assert manifest["input_feedback_file"]["sha256"]
    assert manifest["row_counts"]["validated_feedback_rows"] == 1

    report = json.loads(artifacts.feedback_validation_report_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == "closed-loop-feedback-validation-report-v1"
    assert report["status"] == "passed"
    assert report["checks"]["candidate_ids"]["missing_selected_candidate_ids"] == ["candidate-002"]


def test_ingestion_rejects_unknown_candidate_id(tmp_path):
    from closed_loop.feedback_ingestion import ExperimentalFeedbackIngestionError, ingest_experimental_feedback

    iteration_dir = _iteration_dir(tmp_path)
    feedback_csv = _write_feedback(tmp_path / "feedback.csv", _feedback_rows(candidate_id="candidate-999"))

    with pytest.raises(ExperimentalFeedbackIngestionError, match="unknown candidate_id"):
        ingest_experimental_feedback(iteration_dir=iteration_dir, feedback_csv=feedback_csv)


def test_ingestion_rejects_candidate_smiles_mismatch(tmp_path):
    from closed_loop.feedback_ingestion import ExperimentalFeedbackIngestionError, ingest_experimental_feedback

    iteration_dir = _iteration_dir(tmp_path)
    feedback_csv = _write_feedback(
        tmp_path / "feedback.csv",
        _feedback_rows(smiles="CCN", data_source="lab-a", notes="independent lab measurement row"),
    )

    with pytest.raises(ExperimentalFeedbackIngestionError, match="selected candidate identity"):
        ingest_experimental_feedback(iteration_dir=iteration_dir, feedback_csv=feedback_csv)


def test_ingestion_rejects_duplicate_candidate_id(tmp_path):
    from closed_loop.feedback_ingestion import ExperimentalFeedbackIngestionError, ingest_experimental_feedback

    iteration_dir = _iteration_dir(tmp_path)
    rows = pd.concat([_feedback_rows(), _feedback_rows()], ignore_index=True)
    feedback_csv = _write_feedback(tmp_path / "feedback.csv", rows)

    with pytest.raises(ExperimentalFeedbackIngestionError, match="duplicate candidate_id"):
        ingest_experimental_feedback(
            iteration_dir=iteration_dir,
            feedback_csv=feedback_csv,
            feedback_data_scope="unit_test_fixture",
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("protocol_reference", "", "protocol_reference"),
        ("lab_or_operator", "", "lab_or_operator"),
        ("instrument_or_batch_id", "", "instrument_or_batch_id"),
        ("raw_data_reference", "", "raw_data_reference"),
        ("data_source", "", "data_source"),
        ("measurement_date", "20-05-2026", "measurement_date"),
        ("measured_delta_pce", "nan", "finite numeric"),
        ("measured_pce_with_modulator", "inf", "finite numeric"),
        ("measured_pce_without_modulator", "-0.1", "between 0 and 100"),
        ("measured_pce_with_modulator", "101", "between 0 and 100"),
    ],
)
def test_ingestion_rejects_invalid_required_feedback_fields(
    tmp_path,
    column: str,
    value: str,
    message: str,
):
    from closed_loop.feedback_ingestion import ExperimentalFeedbackIngestionError, ingest_experimental_feedback

    iteration_dir = _iteration_dir(tmp_path)
    feedback_csv = _write_feedback(tmp_path / "feedback.csv", _feedback_rows(**{column: value}))

    with pytest.raises(ExperimentalFeedbackIngestionError, match=message):
        ingest_experimental_feedback(
            iteration_dir=iteration_dir,
            feedback_csv=feedback_csv,
            feedback_data_scope="unit_test_fixture",
        )


def test_ingestion_rejects_delta_that_does_not_match_reported_pces(tmp_path):
    from closed_loop.feedback_ingestion import ExperimentalFeedbackIngestionError, ingest_experimental_feedback

    iteration_dir = _iteration_dir(tmp_path)
    feedback_csv = _write_feedback(tmp_path / "feedback.csv", _feedback_rows(measured_delta_pce=1.5))

    with pytest.raises(ExperimentalFeedbackIngestionError, match="measured_delta_pce"):
        ingest_experimental_feedback(iteration_dir=iteration_dir, feedback_csv=feedback_csv)


def test_ingestion_rejects_fixture_markers_in_external_lab_mode(tmp_path):
    from closed_loop.feedback_ingestion import ExperimentalFeedbackIngestionError, ingest_experimental_feedback

    iteration_dir = _iteration_dir(tmp_path)
    feedback_csv = _write_feedback(
        tmp_path / "synthetic-feedback.csv",
        _feedback_rows(data_source="lab-a", notes="synthetic parser fixture"),
    )

    with pytest.raises(ExperimentalFeedbackIngestionError, match="fixture"):
        ingest_experimental_feedback(
            iteration_dir=iteration_dir,
            feedback_csv=feedback_csv,
            feedback_data_scope="external_lab",
        )
