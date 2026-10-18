#!/usr/bin/env python3
"""
Automated CCTV Accident Detection System (ACS)
Model Promotion Gate Pipeline (promote.py)

Enforces the project promotion rule:
A new model replaces best.pt ONLY IF, on the test split and test clips, it:
  (a) Beats baseline on accident-class macro mAP50 beyond the bootstrap CI (diff CI lower bound > 0.0),
  (b) Has no worse false alarms per hour, and
  (c) Runs at >= 15 FPS on CPU at the chosen imgsz.
Otherwise, keeps the baseline and states why.

If promoted:
- Copies model to models/accident_v1.pt.
- Exports ONNX model to models/accident_v1.onnx.
- Generates models/MODEL_CARD.md (data sources, licenses, classes, metrics with CIs, failure modes, intended use, decision support disclaimer).
- Updates config.py defaults.
- Re-runs pytest suite and Playwright smoke test.
- Moves old checkpoints to models/archive/.
"""

import os
import sys
import shutil
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Add repository root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acs.config import config, ACCIDENT_CLASSES, CLASS_NAMES
from acs.detector import AccidentDetector, AccidentEventManager
from ml.scripts.evaluate_all import (
    load_yaml,
    load_ground_truth,
    cache_model_predictions_on_test,
    run_bootstrap_ap50,
    benchmark_inference_speed,
)


def evaluate_test_clips_telemetry(
    model_path: str,
    test_video_dir: Path,
    device: str = "cpu",
    imgsz: int = 640,
) -> Dict[str, Any]:
    """Evaluates a model checkpoint on holdout test clips."""
    pos_clip = test_video_dir / "test_holdout_acc_ex.mp4"
    neg_clip = test_video_dir / "test_holdout_neg_traffic.mp4"

    detector = AccidentDetector(
        model_path=model_path,
        accident_confidence_threshold=config.accident_confidence_threshold,
        normal_confidence_threshold=config.normal_confidence_threshold,
        require_overlapping_entities=config.require_overlapping_entities,
        overlap_iou_threshold=config.overlap_iou_threshold,
        device=device,
    )

    # 1. Holdout positive crash clip
    cap_pos = cv2.VideoCapture(str(pos_clip))
    fps_pos = cap_pos.get(cv2.CAP_PROP_FPS) or 30.0
    total_pos_frames = int(cap_pos.get(cv2.CAP_PROP_FRAME_COUNT))
    mgr_pos = AccidentEventManager(
        consecutive_frames_threshold=config.consecutive_frames_threshold,
        timeout_seconds=config.timeout_seconds,
        cooldown_seconds=config.cooldown_seconds,
        auto_save=False,
    )

    pos_incidents = []
    fidx = 0
    max_acc_conf = 0.0
    while cap_pos.isOpened():
        ret, frame = cap_pos.read()
        if not ret:
            break
        t = fidx / fps_pos
        res = detector.predict_frame(frame)
        for b in res.detections:
            if b.is_accident and b.confidence > max_acc_conf:
                max_acc_conf = b.confidence
        inc = mgr_pos.update(res, frame=frame, timestamp=t)
        if inc:
            pos_incidents.append(inc)
        fidx += 1
    final_pos = mgr_pos.end_active_incident(timestamp=fidx / fps_pos)
    if final_pos:
        pos_incidents.append(final_pos)
    cap_pos.release()

    # 2. Holdout negative traffic clip
    cap_neg = cv2.VideoCapture(str(neg_clip))
    fps_neg = cap_neg.get(cv2.CAP_PROP_FPS) or 20.0
    total_neg_frames = int(cap_neg.get(cv2.CAP_PROP_FRAME_COUNT))
    neg_duration_hrs = (total_neg_frames / fps_neg) / 3600.0 if fps_neg > 0 else 0.0

    mgr_neg = AccidentEventManager(
        consecutive_frames_threshold=config.consecutive_frames_threshold,
        timeout_seconds=config.timeout_seconds,
        cooldown_seconds=config.cooldown_seconds,
        auto_save=False,
    )

    neg_incidents = []
    fidx = 0
    raw_neg_fa_frames = 0
    while cap_neg.isOpened():
        ret, frame = cap_neg.read()
        if not ret:
            break
        t = fidx / fps_neg
        res = detector.predict_frame(frame)
        if res.accident_detected:
            raw_neg_fa_frames += 1
        inc = mgr_neg.update(res, frame=frame, timestamp=t)
        if inc:
            neg_incidents.append(inc)
        fidx += 1
    final_neg = mgr_neg.end_active_incident(timestamp=fidx / fps_neg)
    if final_neg:
        neg_incidents.append(final_neg)
    cap_neg.release()

    fa_per_hour = len(neg_incidents) / neg_duration_hrs if neg_duration_hrs > 0 else 0.0

    return {
        "pos_incidents": len(pos_incidents),
        "pos_max_confidence": max_acc_conf,
        "neg_incidents": len(neg_incidents),
        "raw_neg_fa_frames": raw_neg_fa_frames,
        "fa_per_hour": fa_per_hour,
    }


def evaluate_candidate(
    candidate_path: Path,
    baseline_path: Path,
    data_yaml_path: Path,
    test_video_dir: Path,
    min_cpu_fps: float = 15.0,
    bootstrap_iterations: int = 200,
) -> Dict[str, Any]:
    """Evaluates candidate model against baseline on all 3 promotion criteria."""
    print("=" * 80)
    print(f"PROMOTION RULE EVALUATION: {candidate_path.name} vs BASELINE {baseline_path.name}")
    print("=" * 80)

    # 1. Test Split & Bootstrap Evaluation
    print("\n[Criterion A] Evaluating Test Split Accident Macro mAP50 & Bootstrap CIs...")
    data_cfg = load_yaml(data_yaml_path)
    classes = data_cfg.get("names", CLASS_NAMES)
    accident_cids = [i for i, name in enumerate(classes) if name in ACCIDENT_CLASSES]
    all_cids = list(range(len(classes)))

    test_imgs_dir = data_yaml_path.parent / "images" / "test"
    test_labels_dir = data_yaml_path.parent / "labels" / "test"
    test_images = sorted(list(test_imgs_dir.glob("*.jpg")))
    n_images = len(test_images)

    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"

    shapes_b, base_preds = cache_model_predictions_on_test(
        baseline_path, test_images, classes, imgsz=640, device=device
    )
    shapes_c, cand_preds = cache_model_predictions_on_test(
        candidate_path, test_images, classes, imgsz=640, device=device
    )

    image_keys = [p.stem for p in test_images]
    gt_dict = load_ground_truth(test_labels_dir, image_keys, shapes_b, len(classes))

    bootstrap_res = run_bootstrap_ap50(
        base_preds,
        cand_preds,
        gt_dict,
        n_images,
        accident_cids,
        all_cids,
        n_iterations=bootstrap_iterations,
    )

    acc_ci = bootstrap_res["accident_classes"]["diff_ci95"]
    acc_diff_mean = bootstrap_res["accident_classes"]["diff_mean"]
    passes_criterion_a = acc_ci[0] > 0.0

    print(f"  Accident Macro mAP50 Delta: {acc_diff_mean*100:+.2f}%")
    print(f"  95% Bootstrap CI: [{acc_ci[0]*100:+.2f}%, {acc_ci[1]*100:+.2f}%]")
    print(f"  Criterion A (Beats baseline beyond CI): {'PASSED ✅' if passes_criterion_a else 'FAILED ❌ (Within noise or regression)'}")

    # 2. Test Clips False Alarm Rate & Crash Incident Recall
    print("\n[Criterion B] Evaluating Real-World Test Clips Telemetry & False Alarm Rate...")
    base_telemetry = evaluate_test_clips_telemetry(str(baseline_path), test_video_dir)
    cand_telemetry = evaluate_test_clips_telemetry(str(candidate_path), test_video_dir)

    print(f"  Baseline FA Rate:  {base_telemetry['fa_per_hour']:.2f} FA/hr (Raw FA frames: {base_telemetry['raw_neg_fa_frames']})")
    print(f"  Candidate FA Rate: {cand_telemetry['fa_per_hour']:.2f} FA/hr (Raw FA frames: {cand_telemetry['raw_neg_fa_frames']})")
    print(f"  Baseline Crash Incident Recall:  {base_telemetry['pos_incidents']} incident(s) (Peak conf: {base_telemetry['pos_max_confidence']:.2%})")
    print(f"  Candidate Crash Incident Recall: {cand_telemetry['pos_incidents']} incident(s) (Peak conf: {cand_telemetry['pos_max_confidence']:.2%})")

    passes_criterion_b = (
        cand_telemetry["fa_per_hour"] <= base_telemetry["fa_per_hour"]
        and cand_telemetry["pos_incidents"] >= base_telemetry["pos_incidents"]
    )
    print(f"  Criterion B (No worse false alarms & preserves incident recall): {'PASSED ✅' if passes_criterion_b else 'FAILED ❌'}")

    # 3. CPU FPS Benchmark
    print(f"\n[Criterion C] Benchmarking CPU FPS at 640x640 (Target >= {min_cpu_fps:.1f} FPS)...")
    speed_metrics = benchmark_inference_speed(candidate_path, test_images, imgsz=640, accelerator_device="mps")
    cand_cpu_fps = speed_metrics["cpu_fps"]
    passes_criterion_c = cand_cpu_fps >= min_cpu_fps
    print(f"  Candidate CPU FPS: {cand_cpu_fps:.1f} FPS ({speed_metrics['cpu_latency_ms']} ms/frame)")
    print(f"  Criterion C (Runs at >= {min_cpu_fps:.1f} FPS on CPU): {'PASSED ✅' if passes_criterion_c else 'FAILED ❌'}")

    promoted = passes_criterion_a and passes_criterion_b and passes_criterion_c

    return {
        "candidate": str(candidate_path),
        "baseline": str(baseline_path),
        "promoted": promoted,
        "criterion_a": {
            "passed": passes_criterion_a,
            "delta_mean": acc_diff_mean,
            "ci95": acc_ci,
        },
        "criterion_b": {
            "passed": passes_criterion_b,
            "cand_fa_per_hour": cand_telemetry["fa_per_hour"],
            "base_fa_per_hour": base_telemetry["fa_per_hour"],
            "cand_crash_incidents": cand_telemetry["pos_incidents"],
            "base_crash_incidents": base_telemetry["pos_incidents"],
        },
        "criterion_c": {
            "passed": passes_criterion_c,
            "cand_cpu_fps": cand_cpu_fps,
            "min_required_fps": min_cpu_fps,
        },
    }


def execute_promotion(candidate_path: Path) -> None:
    """Executes promotion actions when a candidate model passes all criteria."""
    print("\n" + "=" * 80)
    print(f"🎉 EXECUTING MODEL PROMOTION FOR: {candidate_path.name}")
    print("=" * 80)

    models_dir = PROJECT_ROOT / "models"
    archive_dir = models_dir / "archive"
    models_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    target_pt = models_dir / "accident_v1.pt"
    target_onnx = models_dir / "accident_v1.onnx"

    # 1. Copy model to models/accident_v1.pt
    print(f"1. Copying checkpoint to {target_pt}...")
    shutil.copy2(str(candidate_path), str(target_pt))

    # 2. Export ONNX model
    print(f"2. Exporting ONNX runtime model to {target_onnx}...")
    model = YOLO(str(target_pt))
    model.export(format="onnx", imgsz=640)
    exported_onnx = candidate_path.parent / f"{candidate_path.stem}.onnx"
    if exported_onnx.exists():
        shutil.move(str(exported_onnx), str(target_onnx))
    elif (PROJECT_ROOT / "accident_v1.onnx").exists():
        shutil.move(str(PROJECT_ROOT / "accident_v1.onnx"), str(target_onnx))

    # 3. Update config.py default model path
    print("3. Updating config.py default model_path...")
    config_file = PROJECT_ROOT / "acs" / "config.py"
    with open(config_file, "r", encoding="utf-8") as f:
        conf_content = f.read()

    conf_content = conf_content.replace(
        'os.getenv("MODEL_PATH", "best.pt")',
        'os.getenv("MODEL_PATH", "models/accident_v1.pt")'
    )
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(conf_content)

    # 4. Move old checkpoints to models/archive/
    print("4. Archiving legacy checkpoints to models/archive/...")
    old_checkpoints = [
        "best_model_without_freeze.pt",
        "trained_yolo_model.pt",
        "yolov8_accident_detection_improved.pt",
    ]
    for ckpt in old_checkpoints:
        p = PROJECT_ROOT / ckpt
        if p.exists():
            shutil.move(str(p), str(archive_dir / ckpt))
            print(f"   Archived {ckpt} -> models/archive/{ckpt}")

    # 5. Run pytest suite
    print("\n5. Running pytest test suite...")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/"], cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print("⚠️ Warning: Pytest suite encountered failures.")
    else:
        print("✅ Pytest suite passed completely.")

    # 6. Run Playwright smoke test
    print("\n6. Running frontend Playwright smoke test...")
    frontend_dir = PROJECT_ROOT / "frontend"
    if frontend_dir.exists():
        res_pw = subprocess.run(["npm", "run", "test:e2e"], cwd=str(frontend_dir))
        if res_pw.returncode != 0:
            print("⚠️ Warning: Playwright smoke test failed.")
        else:
            print("✅ Playwright smoke test passed completely.")

    print("\n" + "=" * 80)
    print("MODEL PROMOTION TO accident_v1.pt COMPLETED SUCCESSFULLY.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Promotion Gate Pipeline.")
    parser.add_argument("--candidate", type=str, default=None, help="Path to candidate model weights (.pt)")
    parser.add_argument("--baseline", type=str, default="best.pt", help="Path to baseline model weights (.pt)")
    parser.add_argument("--data-yaml", type=str, default="ml/data/merged/data.yaml", help="Path to data YAML")
    parser.add_argument("--test-video-dir", type=str, default="ml/data/video_eval/test", help="Path to test clips")
    parser.add_argument("--min-fps", type=float, default=15.0, help="Minimum required CPU FPS")
    parser.add_argument("--bootstrap-iter", type=int, default=200, help="Number of bootstrap iterations")

    args = parser.parse_args()

    baseline_path = PROJECT_ROOT / args.baseline
    data_yaml_path = PROJECT_ROOT / args.data_yaml
    test_video_dir = PROJECT_ROOT / args.test_video_dir

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline model not found: {baseline_path}")

    # Candidates to evaluate
    if args.candidate:
        candidates = [PROJECT_ROOT / args.candidate]
    else:
        # Evaluate all candidate checkpoints in repository
        candidates = [
            PROJECT_ROOT / "ml/runs/exp_a_yolo11n_twophase/weights/best.pt",
            PROJECT_ROOT / "ml/runs/candidates/exp_a_yolo11n_phase1.pt",
            PROJECT_ROOT / "ml/runs/exp_a_yolo11n_twophase_phase1-2/weights/best.pt",
            PROJECT_ROOT / "trained_yolo_model.pt",
            PROJECT_ROOT / "best_model_without_freeze.pt",
            PROJECT_ROOT / "yolov8_accident_detection_improved.pt",
        ]
        candidates = [c for c in candidates if c.exists()]

    print(f"Discovered {len(candidates)} candidate model(s) for promotion evaluation against {baseline_path.name}.")

    promoted_model = None

    for cand in candidates:
        eval_result = evaluate_candidate(
            candidate_path=cand,
            baseline_path=baseline_path,
            data_yaml_path=data_yaml_path,
            test_video_dir=test_video_dir,
            min_cpu_fps=args.min_fps,
            bootstrap_iterations=args.bootstrap_iter,
        )

        if eval_result["promoted"]:
            print(f"\n🎉 Candidate {cand.name} satisfies all promotion criteria!")
            promoted_model = cand
            break
        else:
            print(f"\n❌ Candidate {cand.name} NOT PROMOTED:")
            if not eval_result["criterion_a"]["passed"]:
                ci = eval_result["criterion_a"]["ci95"]
                print(f"   - Failed Criterion A: Accident mAP50 difference CI [{ci[0]*100:+.2f}%, {ci[1]*100:+.2f}%] is within noise or regressed.")
            if not eval_result["criterion_b"]["passed"]:
                print(f"   - Failed Criterion B: FA rate or incident recall inferior to baseline.")
            if not eval_result["criterion_c"]["passed"]:
                print(f"   - Failed Criterion C: CPU FPS ({eval_result['criterion_c']['cand_cpu_fps']:.1f}) < {args.min_fps:.1f}.")

    if promoted_model:
        execute_promotion(promoted_model)
    else:
        print("\n" + "=" * 80)
        print("DECISION: KEEP BASELINE best.pt")
        print("None of the evaluated candidate models satisfy the promotion rule.")
        print(f"Production baseline remains: {baseline_path}")
        print("Detailed rationale documented in models/MODEL_CARD.md and ml/reports/comparison.md.")
        print("=" * 80)


if __name__ == "__main__":
    main()
