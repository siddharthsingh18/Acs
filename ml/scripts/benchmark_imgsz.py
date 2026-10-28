#!/usr/bin/env python3
"""
ml/scripts/benchmark_imgsz.py
──────────────────────────────────────────────────────────────────────────────
Benchmarks imgsz 640, 960, 1280 on the *small-object slice* from the frozen
test split (from slice_analysis.py – boxes with max-side < 32 px on a 640-wide
frame).  Reports:
  • Accident Recall      – fraction of images with ≥1 GT accident box where the
                           model fires an accident detection.
  • False Alarms / Hour  – spurious accident detections / normalised video hour.
  • Throughput (FPS)     – measured on the same image set, no data-loader overhead.

Decision rule: adopt the next resolution step ONLY IF
  (a) Δrecall > +5 pp   AND  (b) FPS ≥ FPS_TARGET (default 10).

Usage:
  ./venv/bin/python3 ml/scripts/benchmark_imgsz.py [--model best.pt] [--fps-target 10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


ACCIDENT_CLASS_IDS = {1, 2, 3, 5, 6, 7, 8}  # matches slice_analysis.py
SMALL_PX_THRESHOLD = 32            # max side in pixels at imgsz=640 to qualify as "small"
FPS_WARMUP_FRAMES = 5
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_small_object_slice(
    test_img_dir: Path,
    test_lbl_dir: Path,
    small_px: int = SMALL_PX_THRESHOLD,
    reference_imgsz: int = 640,
) -> Tuple[List[Path], List[bool]]:
    """Returns (image_paths, has_accident_flag) for the small-object slice.

    "Small" = at least one GT box whose longest side, when rendered at
    *reference_imgsz* wide, would be < small_px pixels.
    """
    img_paths: List[Path] = []
    has_accident: List[bool] = []

    for img_p in sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))):
        lbl_p = test_lbl_dir / (img_p.stem + ".txt")
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = reference_imgsz / max(w, 1)

        is_small = False
        is_acc = False

        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                bw_norm, bh_norm = float(parts[3]), float(parts[4])
                # Convert to pixel size at reference resolution
                bw_px = bw_norm * w * scale
                bh_px = bh_norm * h * scale
                max_side = max(bw_px, bh_px)
                if max_side < small_px:
                    is_small = True
                if cid in ACCIDENT_CLASS_IDS:
                    is_acc = True

        if is_small:
            img_paths.append(img_p)
            has_accident.append(is_acc)

    return img_paths, has_accident


def _run_benchmark(
    model: YOLO,
    img_paths: List[Path],
    has_accident: List[bool],
    imgsz: int,
    device: str,
    conf: float = 0.35,
) -> Dict[str, Any]:
    """Runs inference on all images and returns metrics dict."""
    tp = 0     # true positives (accident image + model fires accident)
    fn = 0     # false negatives (accident image + model misses)
    fp_frames = 0  # false alarm frames (non-accident image + model fires accident)
    total_ms = 0.0

    acc_img_count = sum(1 for a in has_accident if a)

    for img_p, is_acc_gt in zip(img_paths, has_accident):
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        t0 = time.perf_counter()
        results = model.predict(
            source=img,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        t1 = time.perf_counter()
        total_ms += (t1 - t0) * 1000.0

        model_fired_accident = False
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cid = int(box.cls[0].item())
                if cid in ACCIDENT_CLASS_IDS:
                    model_fired_accident = True
                    break

        if is_acc_gt:
            if model_fired_accident:
                tp += 1
            else:
                fn += 1
        else:
            if model_fired_accident:
                fp_frames += 1

    n = len(img_paths)
    mean_ms = total_ms / max(n, 1)
    fps = 1000.0 / max(mean_ms, 1e-6)

    recall = (tp / acc_img_count) if acc_img_count > 0 else None
    # Normalise FA/hr: assume 25 FPS in a hypothetical stream
    total_non_acc = n - acc_img_count
    stream_hours = (total_non_acc / 25.0) / 3600.0
    fa_per_hour = (fp_frames / stream_hours) if stream_hours > 0 else 0.0

    return {
        "imgsz": imgsz,
        "n_images": n,
        "acc_images": acc_img_count,
        "tp": tp,
        "fn": fn,
        "fp_frames": fp_frames,
        "recall": round(recall, 4) if recall is not None else None,
        "false_alarms_per_hour": round(fa_per_hour, 2),
        "mean_latency_ms": round(mean_ms, 2),
        "fps": round(fps, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best.pt")
    parser.add_argument("--fps-target", type=float, default=10.0,
                        help="Minimum acceptable FPS for adoption.")
    parser.add_argument("--delta-recall-threshold", type=float, default=5.0,
                        help="Minimum recall gain (pp) to justify next resolution step.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument(
        "--test-img-dir",
        type=str,
        default=str(PROJECT_ROOT / "ml" / "data" / "merged" / "images" / "test"),
    )
    parser.add_argument(
        "--test-lbl-dir",
        type=str,
        default=str(PROJECT_ROOT / "ml" / "data" / "merged" / "labels" / "test"),
    )
    args = parser.parse_args()

    device = _resolve_device()
    print("=" * 70)
    print("ACS IMAGE-SIZE BENCHMARK  –  SMALL-OBJECT SLICE")
    print("=" * 70)
    print(f"Model  : {args.model}")
    print(f"Device : {device}")
    print(f"FPS target  : {args.fps_target}")
    print(f"ΔRecall min : {args.delta_recall_threshold} pp")

    # Load test slice
    img_dir = Path(args.test_img_dir)
    lbl_dir = Path(args.test_lbl_dir)
    img_paths, has_accident = _load_small_object_slice(img_dir, lbl_dir)

    if len(img_paths) == 0:
        print("\n⚠️  No small-object images found in test split. Exiting.")
        sys.exit(0)

    print(f"\nSmall-object slice: {len(img_paths)} images "
          f"({sum(has_accident)} with ≥1 accident GT box)\n")

    # Load model once (weights shared across imgsz)
    model = YOLO(args.model)

    # Warm-up at 640
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(FPS_WARMUP_FRAMES):
        model.predict(source=dummy, imgsz=640, conf=0.35, device=device, verbose=False)

    results: List[Dict[str, Any]] = []
    for imgsz in [640, 960, 1280]:
        print(f"Running imgsz={imgsz} ...", end=" ", flush=True)
        r = _run_benchmark(model, img_paths, has_accident, imgsz, device, args.conf)
        results.append(r)
        recall_str = f"{r['recall']*100:.1f}%" if r["recall"] is not None else "N/A (no acc)"
        print(f"Recall={recall_str}  FA/hr={r['false_alarms_per_hour']:.0f}  FPS={r['fps']:.1f}")

    # Decision rule
    print("\n" + "=" * 70)
    print("DECISION ANALYSIS")
    print("=" * 70)

    baseline = results[0]
    recommendation = "KEEP imgsz=640 (baseline already meets requirements)"
    recommended_imgsz = 640

    for r in results[1:]:
        baseline_recall = baseline["recall"] or 0.0
        r_recall = r["recall"] or 0.0
        delta_pp = (r_recall - baseline_recall) * 100.0
        meets_fps = r["fps"] >= args.fps_target
        meets_recall = delta_pp >= args.delta_recall_threshold

        status = "✅ ADOPT" if (meets_fps and meets_recall) else "❌ REJECT"
        print(
            f"imgsz={r['imgsz']}: Δrecall={delta_pp:+.1f} pp  "
            f"FPS={r['fps']:.1f} (target≥{args.fps_target})  → {status}"
        )
        if meets_fps and meets_recall and r["imgsz"] > recommended_imgsz:
            recommended_imgsz = r["imgsz"]
            recommendation = (
                f"ADOPT imgsz={r['imgsz']}  "
                f"(+{delta_pp:.1f} pp recall, {r['fps']:.1f} FPS ≥ target)"
            )

    print(f"\n📌 RECOMMENDATION: {recommendation}")

    # Print summary table
    print("\n" + "─" * 70)
    print(f"{'imgsz':>8}  {'N':>5}  {'Recall':>8}  {'FA/hr':>10}  {'FPS':>7}  {'ms/frame':>10}")
    print("─" * 70)
    for r in results:
        recall_str = f"{r['recall']*100:.1f}%" if r["recall"] is not None else "N/A"
        print(
            f"{r['imgsz']:>8}  {r['n_images']:>5}  {recall_str:>8}  "
            f"{r['false_alarms_per_hour']:>10.0f}  {r['fps']:>7.1f}  {r['mean_latency_ms']:>10.1f}"
        )
    print("─" * 70)

    # Persist JSON report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "device": device,
        "fps_target": args.fps_target,
        "delta_recall_threshold_pp": args.delta_recall_threshold,
        "small_px_threshold": SMALL_PX_THRESHOLD,
        "slice_n": len(img_paths),
        "slice_n_accident": int(sum(has_accident)),
        "results": results,
        "recommended_imgsz": recommended_imgsz,
        "recommendation": recommendation,
    }
    out = REPORTS_DIR / "imgsz_benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n📄 Report written to: {out}")


if __name__ == "__main__":
    main()
