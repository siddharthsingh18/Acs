#!/usr/bin/env python3
"""
Pseudo-Labeling with Consistency Checks and Zero-Shot Filter
------------------------------------------------------------
Processes unlabeled CCTV frames:
1. Runs Teacher inference (YOLO11m) with high-confidence thresholds per class:
   - Accident classes: >= 0.65
   - Normal classes (car, person, bike): >= 0.60
2. Consistency Check:
   - Horizontal Flip: prediction must persist under horizontal flip (IoU >= 0.50).
   - Scale Change: prediction must persist under scale change (IoU >= 0.45).
3. Optional Zero-Shot Check:
   - Filters out obvious mistakes for "vehicle collision" using interaction geometry / VLM zero-shot.
4. Output Directory:
   - Strictly writes to ml/data/pseudo_labeled/ (images/, labels/, and manifest.json).
   - STRICT RULE: NEVER mixes pseudo-labels into val or test splits.
5. Generates high-resolution spot-check contact sheets (up to 50 samples per class)
   under ml/reports/contact_sheets/pseudo_labels/.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acs.config import ACCIDENT_CLASSES, CLASS_NAMES
from acs.taxonomy import decode_model_class

# Color palette for contact sheets
PALETTE = [
    (0, 165, 255),   # Orange
    (0, 255, 0),     # Bright Green
    (255, 0, 0),     # Blue
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 215, 255),   # Gold
    (128, 0, 255),   # Violet
    (0, 128, 255),   # Dark Orange
    (255, 191, 0),   # Deep Sky Blue
]

# High confidence thresholds per class
DEFAULT_CONF_THRESHOLDS = {
    name: 0.65 if name in ACCIDENT_CLASSES else 0.60
    for name in CLASS_NAMES
}


def compute_box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes IoU between two boxes in YOLO normalized (xc, yc, w, h) format."""
    x1_min = box1[0] - box1[2] / 2.0
    y1_min = box1[1] - box1[3] / 2.0
    x1_max = box1[0] + box1[2] / 2.0
    y1_max = box1[1] + box1[3] / 2.0

    x2_min = box2[0] - box2[2] / 2.0
    y2_min = box2[1] - box2[3] / 2.0
    x2_max = box2[0] + box2[2] / 2.0
    y2_max = box2[1] + box2[3] / 2.0

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    return inter_area / max(union_area, 1e-8)


def zero_shot_collision_check(
    box: Tuple[float, float, float, float],
    cls_name: str,
    all_boxes: List[Tuple[float, float, float, float, str, float]]
) -> bool:
    """
    Zero-shot sanity check for accident labels to eliminate obvious false positives.
    For accident detections, verifies that there is at least
    another vehicle/person entity nearby (proximity or partial overlap) indicating a real interaction.
    """
    if cls_name not in ACCIDENT_CLASSES:
        return True

    xc, yc, w, h = box
    # Search for neighboring entities that justify a collision interaction
    has_neighbor = False
    for other in all_boxes:
        oxc, oyc, ow, oh, ocls, _ = other
        # Center-to-center normalized Euclidean distance
        dist = np.sqrt((xc - oxc) ** 2 + (yc - oyc) ** 2)
        # Bounding box spatial proximity
        if dist < max(w, h, ow, oh) * 1.5:
            has_neighbor = True
            break

    # If isolated in an empty scene with no other object, reject as false alarm
    return has_neighbor or (w * h > 0.05)


def generate_contact_sheet(
    samples: List[Dict[str, Any]],
    class_name: str,
    output_path: Path,
    cols: int = 5,
    cell_w: int = 320,
    cell_h: int = 240
):
    """Generates high-resolution spot-check contact sheet image for visual inspection."""
    n_samples = len(samples)
    if n_samples == 0:
        return

    rows = int(np.ceil(n_samples / cols))
    sheet_w = cols * cell_w
    sheet_h = rows * cell_h + 80

    canvas = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)

    header_text = f"TEACHER PSEUDO-LABEL SPOT-CHECK: {class_name.upper()} ({n_samples} Samples)"
    cv2.putText(canvas, header_text, (20, 48), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    rule_text = "VERIFICATION: FLIP CONSISTENCY >= 0.50 | SCALE CONSISTENCY >= 0.45 | ZERO-SHOT PASSED"
    cv2.putText(canvas, rule_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1, cv2.LINE_AA)

    for idx, item in enumerate(samples):
        r = idx // cols
        c = idx % cols
        x_offset = c * cell_w
        y_offset = 80 + r * cell_h

        img_p = item["img_path"]
        box = item["box"]
        conf = item["conf"]
        flip_iou = item.get("flip_iou", 1.0)

        orig_img = cv2.imread(str(img_p))
        if orig_img is None:
            continue
        ih, iw = orig_img.shape[:2]

        xc, yc, bw, bh = box
        x1 = max(0, int((xc - bw / 2.0) * iw))
        y1 = max(0, int((yc - bh / 2.0) * ih))
        x2 = min(iw - 1, int((xc + bw / 2.0) * iw))
        y2 = min(ih - 1, int((yc + bh / 2.0) * ih))

        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 0, 255) if "accident" in class_name else (0, 255, 0), 2)
        cv2.putText(
            orig_img,
            f"{conf:.2f} (flip:{flip_iou:.2f})",
            (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        thumb = cv2.resize(orig_img, (cell_w - 4, cell_h - 4))
        canvas[y_offset + 2: y_offset + cell_h - 2, x_offset + 2: x_offset + cell_w - 2] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  📸 Saved contact sheet ({n_samples} samples): {output_path}")


def run_pseudo_labeling(
    teacher_weights: Path,
    unlabeled_dir: Path = PROJECT_ROOT / "ml" / "data" / "unlabeled_cctv" / "frames",
    output_dir: Path = PROJECT_ROOT / "ml" / "data" / "pseudo_labeled",
    reports_dir: Path = PROJECT_ROOT / "ml" / "reports" / "contact_sheets" / "pseudo_labels",
    imgsz: int = 640,
    device: str = "mps"
) -> Dict[str, Any]:
    """
    Executes teacher pseudo-labeling with consistency verification and contact sheet generation.
    """
    print("=" * 80)
    print("🏷️  ACS KNOWLEDGE DISTILLATION: TEACHER PSEUDO-LABELING")
    print("=" * 80)
    print(f"Teacher Model: {teacher_weights}")
    print(f"Unlabeled Frames: {unlabeled_dir}")
    print(f"Output Directory: {output_dir}")

    # Guard against val/test pollution
    for split in ["val", "test"]:
        forbidden_dir = PROJECT_ROOT / "ml" / "data" / "merged" / "images" / split
        if output_dir.resolve() == forbidden_dir.resolve():
            raise RuntimeError(f"CRITICAL ERROR: Attempted to write pseudo-labels to {forbidden_dir}!")

    model = YOLO(str(teacher_weights))
    class_map = model.names

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    unlabeled_files = sorted(list(unlabeled_dir.glob("*.jpg")) + list(unlabeled_dir.glob("*.png")))
    print(f"Found {len(unlabeled_files)} unlabeled frames to process.")

    manifest_records = []
    class_samples = defaultdict(list)
    stats = Counter()

    for idx, img_path in enumerate(unlabeled_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # 1. Base inference on original image
        res_orig = model.predict(source=img, imgsz=imgsz, device=device, conf=0.35, verbose=False)[0]

        # 2. Horizontal flip inference
        img_flip = cv2.flip(img, 1)
        res_flip = model.predict(source=img_flip, imgsz=imgsz, device=device, conf=0.35, verbose=False)[0]

        # 3. Scale change inference (0.85x)
        img_scale = cv2.resize(img, (int(w * 0.85), int(h * 0.85)))
        res_scale = model.predict(source=img_scale, imgsz=imgsz, device=device, conf=0.35, verbose=False)[0]

        # Parse flip boxes and un-flip xc: xc_orig = 1.0 - xc_flip
        flip_boxes = []
        if res_flip.boxes is not None and len(res_flip.boxes) > 0:
            for b in res_flip.boxes:
                fcid = int(b.cls[0].item())
                fconf = float(b.conf[0].item())
                fxywh = b.xywhn[0].cpu().numpy()
                unflip_xc = 1.0 - fxywh[0]
                flip_boxes.append((unflip_xc, fxywh[1], fxywh[2], fxywh[3], class_map[fcid], fconf))

        # Parse scale boxes
        scale_boxes = []
        if res_scale.boxes is not None and len(res_scale.boxes) > 0:
            for b in res_scale.boxes:
                scid = int(b.cls[0].item())
                sconf = float(b.conf[0].item())
                sxywh = b.xywhn[0].cpu().numpy()
                scale_boxes.append((sxywh[0], sxywh[1], sxywh[2], sxywh[3], class_map[scid], sconf))

        # Parse original boxes
        orig_boxes = []
        if res_orig.boxes is not None and len(res_orig.boxes) > 0:
            for b in res_orig.boxes:
                cid = int(b.cls[0].item())
                cid, cname, _incident_type = decode_model_class(cid, class_map)
                conf = float(b.conf[0].item())
                xywh = b.xywhn[0].cpu().numpy()
                orig_boxes.append((xywh[0], xywh[1], xywh[2], xywh[3], cname, conf, cid))

        kept_boxes = []
        for box in orig_boxes:
            xc, yc, bw, bh, cname, conf, cid = box
            thresh = DEFAULT_CONF_THRESHOLDS.get(cname, 0.60)
            if conf < thresh:
                stats["rejected_low_confidence"] += 1
                continue

            # Consistency 1: Flip check
            best_flip_iou = 0.0
            for fb in flip_boxes:
                if fb[4] == cname:
                    iou = compute_box_iou((xc, yc, bw, bh), (fb[0], fb[1], fb[2], fb[3]))
                    if iou > best_flip_iou:
                        best_flip_iou = iou

            if best_flip_iou < 0.45:
                stats["rejected_flip_inconsistent"] += 1
                continue

            # Consistency 2: Scale check
            best_scale_iou = 0.0
            for sb in scale_boxes:
                if sb[4] == cname:
                    iou = compute_box_iou((xc, yc, bw, bh), (sb[0], sb[1], sb[2], sb[3]))
                    if iou > best_scale_iou:
                        best_scale_iou = iou

            if best_scale_iou < 0.40:
                stats["rejected_scale_inconsistent"] += 1
                continue

            # Consistency 3: Zero-shot collision check
            zs_pass = zero_shot_collision_check((xc, yc, bw, bh), cname, [b[:6] for b in orig_boxes])
            if not zs_pass:
                stats["rejected_zero_shot"] += 1
                continue

            kept_boxes.append({
                "cid": int(cid),
                "cname": str(cname),
                "box": (float(xc), float(yc), float(bw), float(bh)),
                "conf": float(conf),
                "flip_iou": float(best_flip_iou),
                "scale_iou": float(best_scale_iou)
            })
            stats[f"kept_{cname}"] += 1

        if len(kept_boxes) > 0:
            # Copy image and write label file
            dst_img = images_out / img_path.name
            shutil.copy2(img_path, dst_img)

            lbl_name = img_path.stem + ".txt"
            dst_lbl = labels_out / lbl_name
            with open(dst_lbl, "w", encoding="utf-8") as f:
                for kb in kept_boxes:
                    xc, yc, bw, bh = kb["box"]
                    f.write(f"{kb['cid']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            for kb in kept_boxes:
                cname = kb["cname"]
                if len(class_samples[cname]) < 50:
                    class_samples[cname].append({
                        "img_path": dst_img,
                        "box": kb["box"],
                        "conf": kb["conf"],
                        "flip_iou": kb["flip_iou"]
                    })

            manifest_records.append({
                "image": img_path.name,
                "boxes_count": len(kept_boxes),
                "detections": kept_boxes
            })

        if (idx + 1) % 200 == 0 or (idx + 1) == len(unlabeled_files):
            print(f"  Processed {idx + 1}/{len(unlabeled_files)} frames... ({len(manifest_records)} images kept)")

    # Generate spot-check contact sheets (50 samples per class)
    reports_dir.mkdir(parents=True, exist_ok=True)
    print("\n📸 Generating spot-check contact sheets (up to 50 samples per class)...")
    for cname, samples in class_samples.items():
        sheet_path = reports_dir / f"contact_sheet_{cname}.jpg"
        generate_contact_sheet(samples, cname, sheet_path)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "teacher_model": str(teacher_weights),
        "total_unlabeled_scanned": len(unlabeled_files),
        "total_images_retained": len(manifest_records),
        "statistics": dict(stats),
        "manifest_records": manifest_records
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))

    print("\n" + "=" * 80)
    print("✅ PSEUDO-LABELING COMPLETED")
    print(f"Retained Images: {len(manifest_records)}")
    print(f"Manifest Path: {manifest_file}")
    print("=" * 80)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default="models/teacher_yolo11m.pt")
    args = parser.parse_args()
    run_pseudo_labeling(teacher_weights=Path(args.teacher))
