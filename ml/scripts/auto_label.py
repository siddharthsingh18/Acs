#!/usr/bin/env python3
"""
Auto-Labeling & Contact Sheet Generation Script for ACS Model Hardening
----------------------------------------------------------------------
Processes extracted video frames and accident candidate datasets:
1. Runs inference using ml/baseline/best_v0.pt.
2. Filters predictions, keeping ONLY detections with confidence >= 0.60.
3. Saves pre-labels to ml/data/pre_labels/ (labels/ and images/) for review.
   (STRICT RULE: Pre-labels are NOT added to training until spot-checked).
4. NEVER touches or auto-labels the val or test splits.
5. Generates high-resolution contact sheets (up to 50 annotated samples per class)
   under ml/reports/contact_sheets/ to facilitate rapid visual eyeball review.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import yaml

# Consistent color palette for drawing annotations (BGR format)
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


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_contact_sheet(
    samples: List[Dict[str, Any]],
    class_name: str,
    output_path: Path,
    cols: int = 5,
    cell_w: int = 320,
    cell_h: int = 240
):
    """
    Assembles a tiled contact sheet image from a list of sample detections
    with annotated bounding boxes and confidence banners.
    """
    n_samples = len(samples)
    if n_samples == 0:
        return

    rows = int(np.ceil(n_samples / cols))
    sheet_w = cols * cell_w
    sheet_h = rows * cell_h + 80  # 80px header banner

    # Dark background for professional contact sheet presentation
    canvas = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)

    # Draw Header Banner
    header_text = f"ACS PRE-LABEL SPOT-CHECK CONTACT SHEET: {class_name.upper()} ({n_samples} Samples, Conf >= 0.60)"
    cv2.putText(canvas, header_text, (20, 48), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    rule_text = "STATUS: PENDING USER SPOT-CHECK. DO NOT ADD TO TRAINING UNTIL VERIFIED."
    cv2.putText(canvas, rule_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

    for idx, item in enumerate(samples):
        r = idx // cols
        c = idx % cols
        x_offset = c * cell_w
        y_offset = 80 + r * cell_h

        img_p = item["img_path"]
        box = item["box"]  # (xc, yc, bw, bh, conf)
        conf = box[4]

        orig_img = cv2.imread(str(img_p))
        if orig_img is None:
            continue
        ih, iw = orig_img.shape[:2]

        # Draw bounding box
        xc, yc, bw, bh = box[0], box[1], box[2], box[3]
        x1 = max(0, int((xc - bw / 2.0) * iw))
        y1 = max(0, int((yc - bh / 2.0) * ih))
        x2 = min(iw - 1, int((xc + bw / 2.0) * iw))
        y2 = min(ih - 1, int((yc + bh / 2.0) * ih))

        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_str = f"{class_name}: {conf*100:.1f}%"
        (tw, th), bl = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        by1 = max(0, y1 - th - bl - 4)
        cv2.rectangle(orig_img, (x1, by1), (x1 + tw + 6, by1 + th + bl + 4), (0, 255, 0), -1)
        cv2.putText(orig_img, label_str, (x1 + 3, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Resize to cell
        cell_img = cv2.resize(orig_img, (cell_w - 6, cell_h - 26), interpolation=cv2.INTER_AREA)

        # Place cell image onto canvas with padding
        canvas[y_offset + 22 : y_offset + cell_h - 4, x_offset + 3 : x_offset + cell_w - 3] = cell_img

        # Cell border & footer banner with index and filename
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + cell_w, y_offset + cell_h), (50, 50, 50), 1)
        footer_text = f"#{idx+1:02d} | {item['dataset'][:14]} | {conf*100:.1f}%"
        cv2.putText(canvas, footer_text, (x_offset + 8, y_offset + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"Generated contact sheet: {output_path} ({n_samples} samples, {sheet_w}x{sheet_h}px)")


def run_auto_labeling(
    model_path: Path,
    candidates_dir: Path,
    pre_labels_dir: Path,
    contact_sheets_dir: Path,
    classes_path: Path,
    conf_threshold: float = 0.60,
    max_samples_per_source: int = 250,
    contact_sheet_size: int = 50,
    seed: int = 42
):
    """Executes model inference on candidate video frames and generates pre-labels + contact sheets."""
    classes_cfg = load_yaml(classes_path)
    target_classes = classes_cfg.get("classes", [])
    target_names = {i: name for i, name in enumerate(target_classes)}

    print(f"Loading baseline detector from {model_path}...")
    model = YOLO(str(model_path))

    pre_img_dir = pre_labels_dir / "images"
    pre_lbl_dir = pre_labels_dir / "labels"
    pre_img_dir.mkdir(parents=True, exist_ok=True)
    pre_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect candidate images
    rng = random.Random(seed)
    candidate_sources = sorted([d for d in candidates_dir.iterdir() if d.is_dir()])

    candidate_images = []
    print("\nCollecting candidate accident frames across sources:")
    for src_dir in candidate_sources:
        files = sorted(list(src_dir.rglob("*.jpg")))
        rng.shuffle(files)
        sampled = files[:max_samples_per_source] if len(files) > max_samples_per_source else files
        print(f"  [{src_dir.name}]: Selected {len(sampled)} / {len(files)} candidate frames.")
        for f in sampled:
            candidate_images.append((src_dir.name, f))

    print(f"Total candidate frames to process: {len(candidate_images)}")

    # 2. Inference & Pre-Label Generation
    print(f"\nRunning baseline inference (Confidence Threshold >= {conf_threshold*100:.0f}%)...")
    t0 = time.time()

    class_detections = defaultdict(list)
    image_pre_labels = []
    detection_counts = Counter()

    for idx, (ds_name, img_p) in enumerate(candidate_images, start=1):
        if idx % 100 == 0 or idx == len(candidate_images):
            print(f"  Progress: {idx}/{len(candidate_images)} frames processed ({time.time()-t0:.1f}s)...")

        results = model.predict(str(img_p), conf=conf_threshold, verbose=False)[0]
        valid_boxes = []

        for box in results.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            cname = target_names.get(cid, model.names.get(cid, f"ID_{cid}"))

            xywhn = box.xywhn[0].cpu().numpy()
            xc, yc, bw, bh = float(xywhn[0]), float(xywhn[1]), float(xywhn[2]), float(xywhn[3])

            valid_boxes.append((cid, xc, yc, bw, bh, conf))
            detection_counts[cname] += 1

            # Store for contact sheet
            class_detections[cname].append({
                "img_path": img_p,
                "dataset": ds_name,
                "box": (xc, yc, bw, bh, conf)
            })

        if valid_boxes:
            clean_stem = f"pre_{ds_name}_{img_p.stem}"
            out_img = pre_img_dir / f"{clean_stem}.jpg"
            out_lbl = pre_lbl_dir / f"{clean_stem}.txt"

            shutil.copy2(img_p, out_img)
            lbl_lines = [f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cid, xc, yc, bw, bh, _ in valid_boxes]
            out_lbl.write_text("\n".join(lbl_lines), encoding="utf-8")

            image_pre_labels.append({
                "source_dataset": ds_name,
                "image_file": str(out_img.name),
                "box_count": len(valid_boxes),
                "boxes": [{"class": target_names[cid], "conf": round(conf, 4)} for cid, _, _, _, _, conf in valid_boxes]
            })

    elapsed = time.time() - t0
    print(f"\nInference completed in {elapsed:.1f}s ({len(candidate_images)/elapsed:.1f} fps).")
    print(f"Total images with pre-labels (conf >= {conf_threshold}): {len(image_pre_labels)}")
    print(f"Total high-confidence bounding boxes generated: {sum(detection_counts.values())}")

    print("\nPre-Label Box Counts by Class:")
    for cid, cname in enumerate(target_classes):
        cnt = detection_counts.get(cname, 0)
        print(f"  Class {cid:2d} ({cname:22s}): {cnt:5d} boxes")

    # Save Manifest
    manifest_path = pre_labels_dir / "manifest.json"
    manifest_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence_threshold": conf_threshold,
        "total_candidate_frames_scanned": len(candidate_images),
        "pre_labeled_images_count": len(image_pre_labels),
        "total_detections_count": sum(detection_counts.values()),
        "detections_per_class": dict(detection_counts),
        "images": image_pre_labels
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved pre-label manifest to {manifest_path}")

    # 3. Generate Contact Sheets (up to 50 samples per class)
    print(f"\nGenerating contact sheets (up to {contact_sheet_size} spot-check samples per class)...")
    contact_sheet_paths = {}

    for cname, samples in class_detections.items():
        # Sort samples by confidence descending, take top contact_sheet_size
        samples.sort(key=lambda x: x["box"][4], reverse=True)
        top_samples = samples[:contact_sheet_size]

        cs_path = contact_sheets_dir / f"contact_sheet_{cname}.jpg"
        build_contact_sheet(
            samples=top_samples,
            class_name=cname,
            output_path=cs_path,
            cols=5,
            cell_w=320,
            cell_h=240
        )
        contact_sheet_paths[cname] = cs_path

    # 4. Generate Contact Sheet Review Guide
    generate_contact_sheet_report(
        contact_sheet_paths=contact_sheet_paths,
        detection_counts=detection_counts,
        target_classes=target_classes,
        conf_threshold=conf_threshold,
        total_pre_labeled_images=len(image_pre_labels),
        report_path=contact_sheets_dir / "README.md"
    )

    return {
        "candidate_frames_scanned": len(candidate_images),
        "pre_labeled_images": len(image_pre_labels),
        "detection_counts": detection_counts,
        "contact_sheets": contact_sheet_paths
    }


def generate_contact_sheet_report(
    contact_sheet_paths: Dict[str, Path],
    detection_counts: Counter,
    target_classes: List[str],
    conf_threshold: float,
    total_pre_labeled_images: int,
    report_path: Path
):
    """Writes a markdown review guide with embedded contact sheet links for eyeball inspection."""
    lines = []
    lines.append("# ACS Pre-Label Spot-Check & Eyeball Review Guide")
    lines.append("## Automated Video Frame Inference (Baseline `best_v0.pt`)\n")
    lines.append(f"**Generation Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Confidence Threshold**: `conf >= {conf_threshold:.2f}`  ")
    lines.append(f"**Pre-Labeled Images Generated**: `{total_pre_labeled_images:,}` images in `ml/data/pre_labels/`\n")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Operational Rule**: Pre-labels in `ml/data/pre_labels/` are strictly quarantined for review and **have NOT been added to the active training set**.")
    lines.append("> The validation (`val`) and test (`test`) splits were **NEVER auto-labeled**.\n")

    lines.append("### 1. Pre-Label Detection Summary by Target Class\n")
    lines.append("| Class ID | Class Name | Category | High-Conf Detections (>=0.60) | Review Status | Contact Sheet |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :--- |")

    for cid, cname in enumerate(target_classes):
        cnt = detection_counts.get(cname, 0)
        cat = "Accident" if "accident" in cname else "Normal"
        cs_p = contact_sheet_paths.get(cname)
        if cs_p:
            cs_link = f"[`contact_sheet_{cname}.jpg`]({cs_p.name})"
            status = "🔍 **Ready for Spot-Check** (50 samples)"
        else:
            cs_link = "—"
            status = "No high-conf predictions"
        lines.append(f"| {cid} | `{cname}` | {cat} | {cnt:,} | {status} | {cs_link} |")

    lines.append("\n---\n")
    lines.append("### 2. Eyeball Inspection Instructions\n")
    lines.append("1. **Review each Contact Sheet**: Open the image file for each class under `ml/reports/contact_sheets/`.")
    lines.append("2. **Verification Criteria**:")
    lines.append("   - Confirm the bounding box accurately encloses the collision event (not merely unaffected background cars).")
    lines.append("   - Verify that collision subtype matches the class definition (e.g., `car_car_accident` shows vehicle impact, `car_object_accident` shows vehicle impact with guardrail/tree/barrier).")
    lines.append("3. **Promotion to Training**: Only after visual spot-check approval should high-precision candidate clusters be promoted to `ml/data/merged/images/train/`.\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved contact sheet review guide to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-label accident video frames with baseline model and generate contact sheets")
    parser.add_argument("--model", type=str, default="ml/baseline/best_v0.pt", help="Path to baseline YOLO weights")
    parser.add_argument("--candidates_dir", type=str, default="ml/data/raw/accident_candidates", help="Path to candidate accident footage")
    parser.add_argument("--pre_labels_dir", type=str, default="ml/data/pre_labels", help="Path to pre-labels quarantine directory")
    parser.add_argument("--contact_sheets_dir", type=str, default="ml/reports/contact_sheets", help="Path to contact sheets directory")
    parser.add_argument("--classes", type=str, default="ml/configs/classes.yaml", help="Path to classes YAML")
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold")
    parser.add_argument("--max_samples_per_source", type=int, default=250, help="Max candidate frames sampled per source dataset")
    parser.add_argument("--contact_sheet_size", type=int, default=50, help="Max samples per class in contact sheet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_auto_labeling(
        model_path=Path(args.model),
        candidates_dir=Path(args.candidates_dir),
        pre_labels_dir=Path(args.pre_labels_dir),
        contact_sheets_dir=Path(args.contact_sheets_dir),
        classes_path=Path(args.classes),
        conf_threshold=args.conf,
        max_samples_per_source=args.max_samples_per_source,
        contact_sheet_size=args.contact_sheet_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
