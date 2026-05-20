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
            {
                "rank": 3,
                "candidate_id": "candidate-003",
                "smiles": "CN",
                "candidate_score": 0.60,
                "predicted_delta_pce": 0.55,
                "verification_status": "verified",
                "verification_sources": '[{"kind": "molecule", "source": "pubchem"}]',
            },
        ]
    )


def test_iteration_packet_writes_external_measurement_placeholders_only(tmp_path):
    from closed_loop.iteration_packet import generate_iteration_packet, verify_iteration_packet

    ranked_csv = tmp_path / "ranked_candidates.csv"
    _ranked_candidates().to_csv(ranked_csv, index=False)
    iteration_dir = tmp_path / "closed_loop" / "iteration_000"

    artifacts = generate_iteration_packet(
        ranked_candidates_csv=ranked_csv,
        output_dir=iteration_dir,
        dataset_id="closed-loop-fixture",
        iteration_index=0,
        top_k=2,
    )

    expected_files = {
        "iteration_manifest.json",
        "selected_candidates.csv",
        "candidate_packet.md",
        "feedback_schema.json",
        "feedback_template.csv",
        "verification_report.json",
    }
    assert {path.name for path in artifacts.required_files} == expected_files
    assert all((iteration_dir / name).is_file() for name in expected_files)

    selected = pd.read_csv(iteration_dir / "selected_candidates.csv")
    assert selected["candidate_id"].tolist() == ["candidate-001", "candidate-002"]
    assert set(selected["candidate_id"]).issubset(set(_ranked_candidates().head(2)["candidate_id"]))
    assert set(selected["verification_status"]) == {"verified"}

    feedback = pd.read_csv(iteration_dir / "feedback_template.csv", keep_default_na=False)
    assert feedback["candidate_id"].tolist() == ["candidate-001", "candidate-002"]
    assert set(feedback["experimental_status"]) == {"pending_external_measurement"}
    for column in ["measured_delta_pce", "measured_pce", "measurement_date", "protocol_reference", "notes"]:
        assert feedback[column].tolist() == ["", ""]

    schema = json.loads((iteration_dir / "feedback_schema.json").read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["experimental_status"]["enum"] == ["pending_external_measurement"]

    manifest = json.loads((iteration_dir / "iteration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "closed-loop-iteration-packet-v1"
    assert manifest["dataset_id"] == "closed-loop-fixture"
    assert manifest["iteration_index"] == 0
    assert manifest["selection_policy"]["source"] == "ranked_candidates_top_k"
    assert manifest["selection_policy"]["top_k"] == 2
    assert manifest["experimental_measurement_policy"]["does_not_generate_or_fill_measurements"] is True
    assert set(manifest["file_hashes"]) == expected_files - {"iteration_manifest.json"}

    report = verify_iteration_packet(iteration_dir, ranked_candidates_csv=ranked_csv, top_k=2)
    assert report["status"] == "passed"
    assert report["checks"]["required_files"]["status"] == "passed"
    assert report["checks"]["top_k_subset"]["status"] == "passed"
    assert report["checks"]["verified_only"]["status"] == "passed"
    assert report["checks"]["measurement_placeholders_empty"]["status"] == "passed"
    assert report["checks"]["hashes_present"]["status"] == "passed"


def test_iteration_packet_verification_rejects_measurement_values(tmp_path):
    from closed_loop.iteration_packet import ClosedLoopPacketError, generate_iteration_packet, verify_iteration_packet

    ranked_csv = tmp_path / "ranked_candidates.csv"
    _ranked_candidates().to_csv(ranked_csv, index=False)
    iteration_dir = tmp_path / "closed_loop" / "iteration_000"
    generate_iteration_packet(
        ranked_candidates_csv=ranked_csv,
        output_dir=iteration_dir,
        dataset_id="measurement-guard-fixture",
        iteration_index=0,
        top_k=2,
    )

    feedback = pd.read_csv(iteration_dir / "feedback_template.csv", keep_default_na=False)
    feedback.loc[0, "measured_delta_pce"] = "1.23"
    feedback.to_csv(iteration_dir / "feedback_template.csv", index=False)

    with pytest.raises(ClosedLoopPacketError) as exc:
        verify_iteration_packet(iteration_dir, ranked_candidates_csv=ranked_csv, top_k=2)

    assert "measured_delta_pce" in str(exc.value)
