#!/usr/bin/env python3
"""Apply evidence-based promotion gates and generate MODEL_CARD.md."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from acs.taxonomy import CLASSES_PATH
from ml.eval.metrics import clopper_pearson_interval
from ml.eval.report_headers import data_manifest_hash, sha256_file


GATES_PATH = ROOT / "ml" / "promotion_gates.yaml"


def evaluate_gates(artifact: dict[str, Any], gates: dict[str, Any]) -> dict[str, bool]:
    thresholds = gates["defaults"]
    metrics = artifact.get("metrics")
    results = {
        "promotion_thresholds_approved": gates.get("thresholds_approved") is True,
        "evaluation_metrics_present": isinstance(metrics, dict),
        "no_leakage_failures": artifact.get("leakage_failures") == [],
        "no_unverified_claims": artifact.get("unverified_claims") == [],
    }
    if not isinstance(metrics, dict):
        results.update(
            {
                "accident_macro_map50_lcb": False,
                "every_accident_class_evaluable": False,
                "incident_recall_lcb": False,
                "false_alarm_upper_bound": False,
                "minimum_test_events": False,
                "minimum_test_camera_scene_diversity": False,
                "minimum_accident_free_camera_hours": False,
                "robustness_limits": False,
            }
        )
        return results

    map_lcb = metrics.get("accident_macro_map50_lower_bound")
    results["accident_macro_map50_lcb"] = (
        isinstance(map_lcb, (int, float))
        and math.isfinite(map_lcb)
        and map_lcb > thresholds["accident_macro_map50_lower_bound"]
    )
    class_results = metrics.get("accident_classes")
    results["every_accident_class_evaluable"] = bool(class_results) and all(
        item.get("evaluable") is True
        and item.get("gt_val", 0) >= thresholds["minimum_accident_ground_truth_boxes_per_class"]
        and item.get("gt_test", 0) >= thresholds["minimum_accident_ground_truth_boxes_per_class"]
        for item in class_results.values()
    )
    recall_lcb = metrics.get("incident_recall_lower_bound")
    results["incident_recall_lcb"] = (
        isinstance(recall_lcb, (int, float))
        and math.isfinite(recall_lcb)
        and recall_lcb > thresholds["incident_recall_lower_bound"]
    )
    fa_upper = metrics.get("false_alarms_per_camera_hour_upper_bound")
    results["false_alarm_upper_bound"] = (
        isinstance(fa_upper, (int, float))
        and math.isfinite(fa_upper)
        and fa_upper <= thresholds["false_alarms_per_camera_hour_upper_bound"]
    )
    results["minimum_test_events"] = (
        metrics.get("test_events", 0) >= thresholds["minimum_test_events"]
    )
    results["minimum_test_camera_scene_diversity"] = (
        metrics.get("test_cameras_or_scenes", 0) >= thresholds["minimum_test_cameras_or_scenes"]
    )
    results["minimum_accident_free_camera_hours"] = (
        metrics.get("test_accident_free_camera_hours", 0)
        >= thresholds["minimum_accident_free_camera_hours"]
    )
    robustness = metrics.get("robustness_drop_upper_bound")
    conditions = ("fog", "rain", "night", "low_bitrate")
    results["robustness_limits"] = (
        isinstance(robustness, dict)
        and all(
            isinstance(robustness.get(condition), (int, float))
            and math.isfinite(robustness[condition])
            and robustness[condition] <= thresholds["maximum_robustness_drop"]
            for condition in conditions
        )
    )
    return results


def required_sample_notes(gates: dict[str, Any]) -> list[str]:
    defaults = gates["defaults"]
    recall = defaults["incident_recall_lower_bound"]
    false_alarm_rate = defaults["false_alarms_per_camera_hour_upper_bound"]
    # With zero false alarms the exact one-sided 95% Poisson upper bound is 2.996/T.
    hours = math.ceil(-math.log(1 - gates["confidence_level"]) / false_alarm_rate)
    minimum_successes = next(
        count
        for count in range(1, 1000)
        if (interval := clopper_pearson_interval(count, count, gates["confidence_level"]))
        and interval[0] >= recall
    )
    event_minimum = defaults["minimum_test_events"]
    successes_at_event_minimum = min(
        count
        for count in range(event_minimum + 1)
        if (interval := clopper_pearson_interval(count, event_minimum, gates["confidence_level"]))
        and interval[0] >= recall
    )
    return [
        f"Recall lower bound target {recall:.2f}: at least {minimum_successes}/{minimum_successes} "
        f"detected events are needed for an exact two-sided {gates['confidence_level']:.0%} "
        f"Clopper-Pearson lower bound to meet the target. At the {event_minimum}-event minimum, "
        f"at least {successes_at_event_minimum}/{event_minimum} must be detected.",
        f"False-alarm upper bound target {false_alarm_rate:.2f}/camera-hour: zero false alarms "
        f"require at least {hours} camera-hours by the exact one-sided 95% Poisson bound "
        "(2.996/T); the 20-hour minimum remains mandatory.",
    ]


def generate_model_card(
    artifact: dict[str, Any],
    gate_results: dict[str, bool],
    gates: dict[str, Any],
    model_path: Path,
    output_path: Path = ROOT / "MODEL_CARD.md",
) -> str:
    passed = bool(gate_results) and all(gate_results.values())
    status = "production" if passed else "research prototype"
    scope = artifact.get("video_scope") or {}
    test_clips = int(scope.get("test_clips", 0))
    test_hours = float(scope.get("test_camera_hours", 0.0))
    metrics = artifact.get("metrics") or {}
    lines = [
        "# ACS Model Card",
        "",
        "## Status and limitations",
        "",
        f"**Status: {status}.**",
        f"Evaluated on {test_clips} frozen test clips and {test_hours:.3f} test camera-hours.",
        "Accident-class mAP is currently unreliable unless the generated run artifacts show an evaluable "
        "confidence bound for every accident class.",
        "Decision support only. A human must verify every alert; no automated emergency dispatch is permitted.",
        "",
        "This card is generated from `run_artifacts.json`; missing metrics fail closed and cannot be "
        "interpreted as zero.",
        "",
        f"- Model: `{model_path}`",
        f"- Model SHA256: `{artifact.get('run_header', {}).get('model_sha256', 'UNAVAILABLE')}`",
        f"- Data manifest SHA256: `{artifact.get('run_header', {}).get('data_manifest_sha256', 'UNAVAILABLE')}`",
        f"- Class file SHA256: `{artifact.get('run_header', {}).get('class_file_sha256', 'UNAVAILABLE')}`",
        f"- Run ID: `{artifact.get('run_header', {}).get('run_id', 'UNAVAILABLE')}`",
        "",
        "## Promotion gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if result else 'FAIL'} |"
        for name, result in gate_results.items()
    )
    lines.extend(["", "## Evaluation evidence", ""])
    if not metrics:
        lines.append("No quantitative evaluation metrics are available for this run.")
    else:
        lines.extend(
            [
                f"- Accident macro mAP@0.50 lower confidence bound: {metrics.get('accident_macro_map50_lower_bound', 'NOT AVAILABLE')}",
                f"- Incident recall lower confidence bound: {metrics.get('incident_recall_lower_bound', 'NOT AVAILABLE')}",
                f"- False-alarm upper bound: {metrics.get('false_alarms_per_camera_hour_upper_bound', 'NOT AVAILABLE')} per camera-hour",
                f"- Test events: {metrics.get('test_events', 'NOT AVAILABLE')}",
                f"- Test accident-free camera-hours: {metrics.get('test_accident_free_camera_hours', 'NOT AVAILABLE')}",
            ]
        )
    lines.extend(["", "## Minimum evidence requirements", ""])
    lines.extend(f"- {note}" for note in required_sample_notes(gates))
    lines.extend(
        [
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-artifacts", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "best.pt")
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=ROOT / "ml" / "eval" / "data_manifest.yaml",
    )
    parser.add_argument("--model-card", type=Path, default=ROOT / "MODEL_CARD.md")
    args = parser.parse_args()
    if args.run_artifacts is None:
        pointer = ROOT / "ml" / "reports" / "latest_run.txt"
        if not pointer.is_file():
            raise SystemExit("No evaluation run exists. Run `python -m ml.eval.run_all` first.")
        run_id = pointer.read_text(encoding="utf-8").strip()
        args.run_artifacts = ROOT / "ml" / "reports" / "runs" / run_id / "run_artifacts.json"
    artifact = json.loads(args.run_artifacts.read_text(encoding="utf-8"))
    gates = yaml.safe_load(GATES_PATH.read_text(encoding="utf-8"))
    expected_hashes = artifact.get("run_header", {})
    current_hashes = {
        "model_sha256": sha256_file(args.model),
        "data_manifest_sha256": data_manifest_hash(args.data_manifest),
        "class_file_sha256": sha256_file(CLASSES_PATH),
    }
    mismatches = [
        name for name, digest in current_hashes.items() if expected_hashes.get(name) != digest
    ]
    if mismatches:
        raise SystemExit(f"Run artifacts are stale for: {', '.join(mismatches)}")
    results = evaluate_gates(artifact, gates)
    status = generate_model_card(artifact, results, gates, args.model, args.model_card)
    decision = {
        "status": status,
        "gate_results": results,
        "model_sha256": current_hashes["model_sha256"],
        "data_manifest_sha256": current_hashes["data_manifest_sha256"],
        "run_id": expected_hashes.get("run_id"),
    }
    decision_path = args.run_artifacts.parent / "promotion_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Promotion status: {status}")
    if status != "production":
        raise SystemExit(f"Promotion denied; see {args.model_card} and {decision_path}.")



def evaluate_candidate(
    candidate_path: Path,
    baseline_path: Path = None,
    data_yaml_path: Path = None,
    test_video_dir: Path = None,
    min_cpu_fps: float = 15.0,
    bootstrap_iterations: int = 200,
) -> dict[str, Any]:
    return {
        "promoted": False,
        "criterion_a": {"passed": False, "ci95": [0.0, 0.0]},
        "criterion_b": {"passed": False},
        "criterion_c": {"passed": False, "cand_cpu_fps": 0.0},
    }


def execute_promotion(candidate_path: Path) -> None:
    pass


if __name__ == "__main__":
    main()
