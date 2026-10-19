#!/usr/bin/env python3
"""
ml/scripts/eval_nightly_regression.py
Nightly CI incident-level regression evaluation on frozen test clips.

Enforces project quality gate:
Evaluates candidate model on frozen holdout CCTV test clips:
1. Negative traffic clip: checks incident-level False Alarm Rate (FAR per hour).
2. Positive crash clip: checks incident recall and peak detection confidence.
Fails CI (exit code 1) if false alarms per hour regress beyond project threshold.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acs.config import config
from acs.detector import AccidentDetector, AccidentEventManager


def evaluate_negative_test_clip(
    detector: AccidentDetector,
    video_path: Path,
) -> Dict[str, Any]:
    """Evaluates accident-free CCTV footage to measure incident-level false alarm rate."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open negative test clip at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps > 0 else 0.0
    duration_hours = duration_seconds / 3600.0

    mgr = AccidentEventManager(
        consecutive_frames_threshold=config.consecutive_frames_threshold,
        timeout_seconds=config.timeout_seconds,
        cooldown_seconds=config.cooldown_seconds,
        auto_save=False,
    )

    incidents = []
    frame_idx = 0
    raw_accident_frames = 0
    max_false_confidence = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        res = detector.predict_frame(frame)
        if res.accident_detected:
            raw_accident_frames += 1
            for d in res.detections:
                if d.is_accident and d.confidence > max_false_confidence:
                    max_false_confidence = d.confidence

        inc = mgr.update(res, frame=frame, timestamp=t)
        if inc:
            incidents.append(inc)
        frame_idx += 1

    final_inc = mgr.end_active_incident(timestamp=frame_idx / fps)
    if final_inc:
        incidents.append(final_inc)

    cap.release()

    fa_per_hour = len(incidents) / duration_hours if duration_hours > 0 else 0.0

    return {
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
        "duration_hours": duration_hours,
        "incident_count": len(incidents),
        "raw_accident_frames": raw_accident_frames,
        "max_false_confidence": max_false_confidence,
        "fa_per_hour": fa_per_hour,
    }


def evaluate_positive_test_clip(
    detector: AccidentDetector,
    video_path: Path,
) -> Dict[str, Any]:
    """Evaluates positive crash CCTV footage to measure collision detection recall."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open positive test clip at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mgr = AccidentEventManager(
        consecutive_frames_threshold=config.consecutive_frames_threshold,
        timeout_seconds=config.timeout_seconds,
        cooldown_seconds=config.cooldown_seconds,
        auto_save=False,
    )

    incidents = []
    frame_idx = 0
    max_confidence = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        res = detector.predict_frame(frame)
        for d in res.detections:
            if d.is_accident and d.confidence > max_confidence:
                max_confidence = d.confidence

        inc = mgr.update(res, frame=frame, timestamp=t)
        if inc:
            incidents.append(inc)
        frame_idx += 1

    final_inc = mgr.end_active_incident(timestamp=frame_idx / fps)
    if final_inc:
        incidents.append(final_inc)

    cap.release()

    return {
        "total_frames": total_frames,
        "incident_count": len(incidents),
        "detected": len(incidents) > 0,
        "max_confidence": max_confidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Nightly Incident-Level False Alarm Regression Gate")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to model checkpoint")
    parser.add_argument(
        "--test-dir",
        type=str,
        default="ml/data/video_eval/test",
        help="Directory containing frozen test clips (test_holdout_acc_ex.mp4 and test_holdout_neg_traffic.mp4)",
    )
    parser.add_argument(
        "--max-fa-per-hour",
        type=float,
        default=0.25,
        help="Ceiling false alarms per hour threshold (default: 0.25, i.e. 1 per 4 hours)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Compute device for inference")
    parser.add_argument("--track-mlflow", action="store_true", default=True, help="Track run in MLflow")
    parser.add_argument("--output-summary", type=str, default=None, help="Markdown summary output path")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"ERROR: Model checkpoint not found at {model_path}")
        sys.exit(1)

    test_dir = Path(args.test_dir)
    neg_clip = test_dir / "test_holdout_neg_traffic.mp4"
    pos_clip = test_dir / "test_holdout_acc_ex.mp4"

    # Fallbacks if holdout test files are named differently
    if not pos_clip.is_file() and Path("acc_ex.mp4").is_file():
        pos_clip = Path("acc_ex.mp4")

    if not neg_clip.is_file():
        print(f"ERROR: Frozen negative traffic clip not found at {neg_clip}")
        sys.exit(1)

    print("=" * 80)
    print("NIGHTLY INCIDENT-LEVEL REGRESSION EVALUATION")
    print(f"Model: {model_path} | Device: {args.device}")
    print(f"Ceiling False Alarm Budget: <= {args.max_fa_per_hour:.2f} false alarms/hour")
    print("=" * 80)

    detector = AccidentDetector(
        model_path=str(model_path),
        accident_confidence_threshold=config.accident_confidence_threshold,
        normal_confidence_threshold=config.normal_confidence_threshold,
        require_overlapping_entities=config.require_overlapping_entities,
        overlap_iou_threshold=config.overlap_iou_threshold,
        device=args.device,
    )

    # 1. Run negative traffic clip
    print(f"\n[1/2] Evaluating False Alarm Rate on frozen negative clip: {neg_clip.name}...")
    t0 = time.time()
    neg_results = evaluate_negative_test_clip(detector, neg_clip)
    neg_duration_s = time.time() - t0
    print(f"      Duration: {neg_results['duration_seconds']:.1f}s footage evaluated in {neg_duration_s:.1f}s")
    print(f"      Spurious Incidents: {neg_results['incident_count']}")
    print(f"      False Alarms / Hour: {neg_results['fa_per_hour']:.2f}")

    # 2. Run positive accident clip
    print(f"\n[2/2] Evaluating Crash Incident Recall on frozen positive clip: {pos_clip.name}...")
    pos_results = evaluate_positive_test_clip(detector, pos_clip)
    print(f"      Accident Detected: {'YES ✅' if pos_results['detected'] else 'NO ❌'}")
    print(f"      Peak Incident Confidence: {pos_results['max_confidence']:.1%}")

    # 3. Check Regression Gate
    fa_rate = neg_results["fa_per_hour"]
    fa_regressed = fa_rate > args.max_fa_per_hour
    recall_regressed = not pos_results["detected"]

    summary_lines = [
        "# Nightly Incident-Level Regression Evaluation Report",
        "",
        f"- **Model**: `{model_path.name}`",
        f"- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Device**: {args.device}",
        "",
        "## Performance on Frozen Holdout Clips",
        "",
        "| Metric | Result | Target Budget | Status |",
        "|---|---|---|---|",
        f"| **Incident False Alarms / Hour** | `{fa_rate:.2f}` FA/hr | `<={args.max_fa_per_hour:.2f}` FA/hr | {'❌ REGRESSED' if fa_regressed else '✅ PASSED'} |",
        f"| **Crash Incident Recall** | {'Detected' if pos_results['detected'] else 'MISSED'} | Required (>=1) | {'❌ MISSED' if recall_regressed else '✅ PASSED'} |",
        f"| **Peak Collision Confidence** | `{pos_results['max_confidence']:.1%}` | N/A | Info |",
        f"| **Negative Test Footage** | `{neg_results['duration_seconds']:.1f}s` ({neg_results['total_frames']} frames) | Holdout Traffic | Info |",
        "",
    ]

    if fa_regressed:
        summary_lines.append(
            f"> [!CAUTION]\n"
            f"> **FALSE ALARM REGRESSION FAILURE**: False alarms per hour ({fa_rate:.2f}) "
            f"exceeded the project ceiling threshold of {args.max_fa_per_hour:.2f}."
        )
    elif recall_regressed:
        summary_lines.append(
            "> [!CAUTION]\n"
            "> **INCIDENT RECALL FAILURE**: The known collision in the holdout clip was missed."
        )
    else:
        summary_lines.append(
            "> [!NOTE]\n"
            "> **NIGHTLY GATE PASSED**: False alarm rate and collision recall meet project standards."
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # Write summary file
    summary_path = args.output_summary or os.getenv("GITHUB_STEP_SUMMARY") or "reports/nightly_regression_report.md"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"\nWritten summary to {summary_path}")
    except Exception as e:
        print(f"Warning writing summary: {e}")

    # 4. Track in MLflow
    if args.track_mlflow:
        try:
            from acs.tracking import tracker
            with tracker.start_run(
                run_name=f"nightly-{model_path.name}-{time.strftime('%Y%m%d')}",
                tags={"job": "nightly-regression-eval", "gate_passed": str(not (fa_regressed or recall_regressed))},
            ):
                tracker.log_dataset_metadata(
                    name="frozen_test_clips",
                    num_samples={"neg_frames": neg_results["total_frames"], "pos_frames": pos_results["total_frames"]},
                    version="holdout_v1",
                )
                tracker.log_params({
                    "model_filename": model_path.name,
                    "max_fa_threshold": args.max_fa_per_hour,
                    "device": args.device,
                })
                tracker.log_metrics({
                    "false_alarms_per_hour": fa_rate,
                    "incident_recall": 1.0 if pos_results["detected"] else 0.0,
                    "peak_positive_confidence": pos_results["max_confidence"],
                    "spurious_incidents_count": neg_results["incident_count"],
                    "gate_passed": 1.0 if not (fa_regressed or recall_regressed) else 0.0,
                })
                tracker.log_artifact(summary_path)
        except Exception as mlflow_err:
            print(f"MLflow tracking warning: {mlflow_err}")

    if fa_regressed or recall_regressed:
        print("\n❌ NIGHTLY EVALUATION GATE FAILED: Quality regression detected.")
        sys.exit(1)

    print("\n✅ NIGHTLY EVALUATION GATE PASSED: Zero false alarm regression.")
    sys.exit(0)


if __name__ == "__main__":
    main()
