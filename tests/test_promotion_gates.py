from copy import deepcopy
from pathlib import Path

import yaml

from ml.scripts.promote import evaluate_gates, generate_model_card


ROOT = Path(__file__).resolve().parents[1]
GATES = yaml.safe_load((ROOT / "ml/promotion_gates.yaml").read_text(encoding="utf-8"))


def passing_artifact():
    return {
        "leakage_failures": [],
        "unverified_claims": [],
        "video_scope": {"test_clips": 65, "test_camera_hours": 22.5},
        "run_header": {
            "run_id": "run-1",
            "model_sha256": "abc",
            "data_manifest_sha256": "def",
            "class_file_sha256": "ghi",
        },
        "metrics": {
            "accident_macro_map50_lower_bound": 0.6,
            "accident_classes": {
                "accident": {"evaluable": True, "gt_val": 45, "gt_test": 50}
            },
            "incident_recall_lower_bound": 0.82,
            "false_alarms_per_camera_hour_upper_bound": 0.2,
            "test_events": 45,
            "test_cameras_or_scenes": 18,
            "test_accident_free_camera_hours": 22,
            "robustness_drop_upper_bound": {
                "fog": 0.05,
                "rain": 0.05,
                "night": 0.08,
                "low_bitrate": 0.04,
            },
        },
    }


def test_all_promotion_gates_require_and_accept_complete_evidence(tmp_path: Path):
    artifact = passing_artifact()
    approved_gates = deepcopy(GATES)
    approved_gates["thresholds_approved"] = True
    results = evaluate_gates(artifact, approved_gates)
    assert all(results.values())
    card = tmp_path / "MODEL_CARD.md"
    status = generate_model_card(artifact, results, approved_gates, Path("best.pt"), card)
    assert status == "production"
    assert "Evaluated on 65 frozen test clips and 22.500 test camera-hours." in card.read_text()


def test_missing_metrics_or_verified_claims_blocks_promotion():
    artifact = passing_artifact()
    artifact["metrics"] = None
    assert not all(evaluate_gates(artifact, GATES).values())

    artifact = passing_artifact()
    artifact["unverified_claims"] = ["historic mAP"]
    assert not all(evaluate_gates(artifact, GATES).values())

    approved_gates = deepcopy(GATES)
    approved_gates["thresholds_approved"] = True
    artifact = passing_artifact()
    assert not all(evaluate_gates(artifact, GATES).values())
    assert all(evaluate_gates(artifact, approved_gates).values())


def test_gate_sample_notes_show_recall_and_camera_hour_requirements():
    from ml.scripts.promote import required_sample_notes

    notes = "\n".join(required_sample_notes(GATES))
    assert "17/17" in notes
    assert "38/40" in notes
    assert "12 camera-hours" in notes
