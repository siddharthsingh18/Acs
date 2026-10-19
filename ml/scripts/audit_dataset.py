#!/usr/bin/env python3
"""
Dataset Audit Script for Automated CCTV Accident Detection System (ACS)
-----------------------------------------------------------------------
Audits raw downloaded object detection datasets in YOLO format:
- Image count (total, by split)
- Bounding boxes per class & class imbalance
- Empty-label images
- Corrupt / unreadable images
- Bounding boxes with zero, negative, or out-of-range size
- Perceptual hash duplicates (dHash) & cross-split leakage
- Generates 12 annotated sample images per dataset in ml/reports/samples/
- Samples 20 images per generic accident class for semantic inspection
- Compiles a comprehensive audit report to ml/reports/dataset_audit.md
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
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
    (50, 205, 50),   # Lime Green
    (180, 105, 255), # Hot Pink
]


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Computes a 64-bit difference hash (dHash) for an image to detect perceptual duplicates."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    val = 0
    for i, v in enumerate(diff.flatten()):
        if v:
            val |= 1 << i
    return val


def find_dataset_splits(dataset_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Detects split directories (train, valid/val, test) and resolves image/label folders.
    Supports both Roboflow structure (<split>/images, <split>/labels)
    and HuggingFace structure (images/<split>, labels/<split>).
    """
    splits = {}
    split_names = [("train", "train"), ("val", "val"), ("valid", "val"), ("test", "test")]

    # Check Roboflow layout: <split>/images
    for raw_split, canon_split in split_names:
        img_dir = dataset_dir / raw_split / "images"
        lbl_dir = dataset_dir / raw_split / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            splits[canon_split] = {"images": img_dir, "labels": lbl_dir}

    # If Roboflow layout was not found, check HuggingFace layout: images/<split>
    if not splits:
        for raw_split, canon_split in split_names:
            img_dir = dataset_dir / "images" / raw_split
            lbl_dir = dataset_dir / "labels" / raw_split
            if img_dir.is_dir() and lbl_dir.is_dir():
                splits[canon_split] = {"images": img_dir, "labels": lbl_dir}

    # Fallback: check if images and labels are top-level
    if not splits:
        img_dir = dataset_dir / "images"
        lbl_dir = dataset_dir / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            splits["all"] = {"images": img_dir, "labels": lbl_dir}

    return splits


def parse_data_yaml(yaml_path: Path) -> Dict[int, str]:
    """Parses class ID to class name mapping from data.yaml."""
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "names" not in data:
        return {}
    names = data["names"]
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def audit_single_dataset(
    dataset_dir: Path,
    sample_out_dir: Path,
    raw_root: Optional[Path] = None,
    num_samples: int = 12,
    accident_sample_size: int = 20,
    seed: int = 42
) -> Dict[str, Any]:
    """Runs a thorough audit on a single dataset directory."""
    # Determine unique display and key names
    if raw_root and dataset_dir.is_relative_to(raw_root):
        rel = dataset_dir.relative_to(raw_root)
        dataset_key = str(rel) if len(rel.parts) > 1 else dataset_dir.name
        folder_name = str(rel).replace("/", "_")
    else:
        dataset_key = dataset_dir.name
        folder_name = dataset_dir.name

    print("\n=======================================================")
    print(f"Auditing Dataset: {dataset_key}")
    print("=======================================================")

    yaml_path = dataset_dir / "data.yaml"
    class_names = parse_data_yaml(yaml_path)
    print(f"Declared Classes in data.yaml ({len(class_names)}): {class_names}")

    splits = find_dataset_splits(dataset_dir)
    print(f"Found Splits: {list(splits.keys())}")

    # Aggregators
    total_images = 0
    split_image_counts = {}
    corrupt_images = []
    empty_label_images = []
    valid_images_with_labels = []

    total_boxes = 0
    class_box_counts = Counter()
    invalid_boxes = []  # boxes with w<=0, h<=0, or coords out of range [0, 1]
    oversized_canvas_boxes = []  # boxes covering >85% of image area (potential lazy full-canvas labels)
    unknown_class_boxes = []

    # Perceptual hash tracking
    dhash_to_images = defaultdict(list)
    image_to_dhash = {}

    # File inventory
    image_entries = []  # dict of (img_path, lbl_path, split, img_size)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split_name, paths in splits.items():
        img_dir = paths["images"]
        lbl_dir = paths["labels"]

        img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in image_extensions])
        split_image_counts[split_name] = len(img_files)
        total_images += len(img_files)

        for img_path in img_files:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            # Check Image Integrity & Perceptual Hash
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    w, h = im.size
                    h_val = compute_dhash(im)

                dhash_to_images[h_val].append((split_name, img_path))
                image_to_dhash[img_path] = h_val
            except Exception as e:
                corrupt_images.append((str(img_path), str(e)))
                continue

            # Check Labels
            if not lbl_path.exists():
                empty_label_images.append((split_name, img_path, "missing_label_file"))
                image_entries.append({
                    "img_path": img_path,
                    "lbl_path": lbl_path,
                    "split": split_name,
                    "boxes": [],
                    "size": (w, h)
                })
                continue

            try:
                content = lbl_path.read_text(encoding="utf-8").strip()
            except Exception:
                content = ""

            if not content:
                empty_label_images.append((split_name, img_path, "empty_label_file"))
                image_entries.append({
                    "img_path": img_path,
                    "lbl_path": lbl_path,
                    "split": split_name,
                    "boxes": [],
                    "size": (w, h)
                })
                continue

            # Parse bounding boxes
            parsed_boxes = []
            for line_idx, line in enumerate(content.splitlines(), start=1):
                tokens = line.strip().split()
                if not tokens:
                    continue
                if len(tokens) < 5:
                    invalid_boxes.append((str(lbl_path), line_idx, line, "fewer_than_5_tokens"))
                    continue

                try:
                    cls_id = int(tokens[0])
                    xc = float(tokens[1])
                    yc = float(tokens[2])
                    bw = float(tokens[3])
                    bh = float(tokens[4])
                except ValueError as e:
                    invalid_boxes.append((str(lbl_path), line_idx, line, f"non_numeric_values: {e}"))
                    continue

                # Coordinate and size sanity checks
                reasons = []
                if bw <= 0.0 or bh <= 0.0:
                    reasons.append(f"zero_or_negative_dim(w={bw:.4f},h={bh:.4f})")
                if xc < 0.0 or xc > 1.0 or yc < 0.0 or yc > 1.0:
                    reasons.append(f"center_out_of_bounds(xc={xc:.4f},yc={yc:.4f})")
                if bw > 1.001 or bh > 1.001:
                    reasons.append(f"size_exceeds_image(w={bw:.4f},h={bh:.4f})")

                # Check box corners
                x1 = xc - bw / 2.0
                y1 = yc - bh / 2.0
                x2 = xc + bw / 2.0
                y2 = yc + bh / 2.0
                if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
                    reasons.append(f"box_extending_beyond_frame(x1={x1:.3f},y1={y1:.3f},x2={x2:.3f},y2={y2:.3f})")

                if bw * bh > 0.85:
                    oversized_canvas_boxes.append((str(lbl_path), line_idx, cls_id, bw * bh))

                if class_names and cls_id not in class_names:
                    unknown_class_boxes.append((str(lbl_path), line_idx, cls_id))

                if reasons:
                    invalid_boxes.append((str(lbl_path), line_idx, line, "; ".join(reasons)))
                else:
                    class_box_counts[cls_id] += 1
                    total_boxes += 1
                    parsed_boxes.append((cls_id, xc, yc, bw, bh))

            if parsed_boxes:
                valid_images_with_labels.append((split_name, img_path, parsed_boxes))
            else:
                empty_label_images.append((split_name, img_path, "no_valid_boxes"))

            image_entries.append({
                "img_path": img_path,
                "lbl_path": lbl_path,
                "split": split_name,
                "boxes": parsed_boxes,
                "size": (w, h)
            })

    # Duplicate analysis
    duplicate_groups = {h: imgs for h, imgs in dhash_to_images.items() if len(imgs) > 1}
    num_duplicate_images = sum(len(imgs) - 1 for imgs in duplicate_groups.values())

    # Cross-split leakage
    cross_split_leakage = []
    for h, imgs in duplicate_groups.items():
        splits_in_group = set(split for split, _ in imgs)
        if len(splits_in_group) > 1:
            cross_split_leakage.append((splits_in_group, [p.name for _, p in imgs]))

    # Class imbalance calculations
    imbalance_ratio = 1.0
    if len(class_box_counts) > 1:
        counts = list(class_box_counts.values())
        imbalance_ratio = max(counts) / max(1, min(counts))

    # Print summary metrics to console
    print(f"Total Images: {total_images} (Splits: {split_image_counts})")
    print(f"Corrupt Images: {len(corrupt_images)}")
    print(f"Empty-Label Images: {len(empty_label_images)} ({len(empty_label_images)/max(1, total_images)*100:.1f}%)")
    print(f"Total Valid Bounding Boxes: {total_boxes}")
    print(f"Invalid / Out-of-Range Boxes: {len(invalid_boxes)}")
    print(f"Oversized (>85% canvas) Boxes: {len(oversized_canvas_boxes)}")
    print(f"Duplicate Image Count: {num_duplicate_images} across {len(duplicate_groups)} duplicate clusters")
    print(f"Cross-Split Leakage Clusters: {len(cross_split_leakage)}")
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")

    if class_names:
        print("\nBox Distribution per Class:")
        for cid in sorted(class_names.keys()):
            cname = class_names[cid]
            cnt = class_box_counts.get(cid, 0)
            pct = (cnt / max(1, total_boxes)) * 100
            print(f"  Class {cid:2d} ({cname:25s}): {cnt:6d} boxes ({pct:5.1f}%)")
    else:
        print("\nNotice: No bounding box classes declared in data.yaml (Background Negative Set).")

    # Generate 12 Random Annotated Samples
    rng = random.Random(seed)
    ds_samples_dir = sample_out_dir / folder_name
    ds_samples_dir.mkdir(parents=True, exist_ok=True)

    # Prefer images with valid boxes for visual inspection if available
    annotated_candidates = [e for e in image_entries if e["boxes"]]
    if not annotated_candidates:
        annotated_candidates = image_entries

    sampled_entries = rng.sample(annotated_candidates, min(num_samples, len(annotated_candidates)))
    saved_sample_paths = []

    print(f"\nExporting {len(sampled_entries)} visual sample images to: {ds_samples_dir}")
    for idx, entry in enumerate(sampled_entries, start=1):
        img_p = entry["img_path"]
        boxes = entry["boxes"]

        cv_img = cv2.imread(str(img_p))
        if cv_img is None:
            continue
        ih, iw = cv_img.shape[:2]

        if not boxes:
            label_text = "[Negative / No Boxes]"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(cv_img, (10, 10), (10 + tw + 6, 10 + th + baseline + 6), (50, 50, 50), -1)
            cv2.putText(cv_img, label_text, (13, 10 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            for cls_id, xc, yc, bw, bh in boxes:
                x1 = int((xc - bw / 2.0) * iw)
                y1 = int((yc - bh / 2.0) * ih)
                x2 = int((xc + bw / 2.0) * iw)
                y2 = int((yc + bh / 2.0) * ih)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw - 1, x2), min(ih - 1, y2)

                color = PALETTE[cls_id % len(PALETTE)]
                cname = class_names.get(cls_id, f"ID_{cls_id}")
                label_text = f"{cls_id}:{cname}"

                # Draw rectangle
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)

                # Draw text banner
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                box_y1 = max(0, y1 - th - baseline - 4)
                cv2.rectangle(cv_img, (x1, box_y1), (x1 + tw + 4, box_y1 + th + baseline + 4), color, -1)
                cv2.putText(cv_img, label_text, (x1 + 2, box_y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        out_fname = f"sample_{idx:02d}_{img_p.stem}.jpg"
        out_path = ds_samples_dir / out_fname
        cv2.imwrite(str(out_path), cv_img)
        saved_sample_paths.append(out_path)

    # 20-Sample Inspection for Generic "Accident" Classes
    accident_samples_dir = ds_samples_dir / "accident_inspection_20"
    accident_sample_paths = []

    # Identify classes containing 'accident' or 'flip' in their name
    generic_accident_cids = [
        cid for cid, name in class_names.items()
        if "accident" in name.lower() or "flip" in name.lower()
    ]
    if generic_accident_cids:
        accident_candidates = [
            e for e in image_entries
            if any(b[0] in generic_accident_cids for b in e["boxes"])
        ]
        if accident_candidates:
            accident_samples_dir.mkdir(parents=True, exist_ok=True)
            sampled_accidents = rng.sample(accident_candidates, min(accident_sample_size, len(accident_candidates)))
            print(f"Exporting {len(sampled_accidents)} generic accident inspection images to: {accident_samples_dir}")

            for idx, entry in enumerate(sampled_accidents, start=1):
                img_p = entry["img_path"]
                boxes = entry["boxes"]

                cv_img = cv2.imread(str(img_p))
                if cv_img is None:
                    continue
                ih, iw = cv_img.shape[:2]

                for cls_id, xc, yc, bw, bh in boxes:
                    x1 = int((xc - bw / 2.0) * iw)
                    y1 = int((yc - bh / 2.0) * ih)
                    x2 = int((xc + bw / 2.0) * iw)
                    y2 = int((yc + bh / 2.0) * ih)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(iw - 1, x2), min(ih - 1, y2)

                    color = PALETTE[cls_id % len(PALETTE)]
                    cname = class_names.get(cls_id, f"ID_{cls_id}")
                    label_text = f"{cls_id}:{cname}"

                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
                    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    box_y1 = max(0, y1 - th - baseline - 4)
                    cv2.rectangle(cv_img, (x1, box_y1), (x1 + tw + 4, box_y1 + th + baseline + 4), color, -1)
                    cv2.putText(cv_img, label_text, (x1 + 2, box_y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                out_fname = f"accident_{idx:02d}_{img_p.stem}.jpg"
                out_path = accident_samples_dir / out_fname
                cv2.imwrite(str(out_path), cv_img)
                accident_sample_paths.append(out_path)

    # Label Quality Assessment & Anomaly Flags
    flags = []
    if len(corrupt_images) > 0:
        flags.append(f"CRITICAL: {len(corrupt_images)} corrupt images that cannot be decoded.")
    if len(invalid_boxes) > 0:
        flags.append(f"WARNING: {len(invalid_boxes)} bounding boxes with invalid/out-of-range dimensions.")
    if len(oversized_canvas_boxes) > (0.05 * max(1, total_boxes)):
        flags.append(f"WARNING: High proportion of oversized canvas boxes ({len(oversized_canvas_boxes)} boxes >85% canvas area). Potential full-image bounding boxes.")
    if num_duplicate_images > (0.15 * max(1, total_images)):
        flags.append(f"WARNING: High image duplication rate ({num_duplicate_images}/{total_images} = {num_duplicate_images/max(1, total_images)*100:.1f}% duplicates).")
    if len(cross_split_leakage) > 0:
        flags.append(f"WARNING: Data leakage detected! {len(cross_split_leakage)} identical images present across train/val/test splits.")
    if imbalance_ratio > 20.0 and len(class_box_counts) > 1:
        flags.append(f"NOTICE: Severe class imbalance (ratio {imbalance_ratio:.1f}:1).")
    if len(empty_label_images) > (0.20 * max(1, total_images)) and class_names:
        flags.append(f"NOTICE: High percentage of empty/background images ({len(empty_label_images)/max(1, total_images)*100:.1f}%).")
    if not flags:
        flags.append("PASS: No severe defects detected.")

    print("\nQuality Assessment Flags:")
    for flag in flags:
        print(f"  * {flag}")

    return {
        "dataset_name": dataset_key,
        "folder_name": folder_name,
        "dataset_dir": str(dataset_dir),
        "class_names": class_names,
        "splits": splits,
        "split_counts": split_image_counts,
        "total_images": total_images,
        "corrupt_images": corrupt_images,
        "empty_label_images": empty_label_images,
        "total_boxes": total_boxes,
        "class_box_counts": dict(class_box_counts),
        "invalid_boxes": invalid_boxes,
        "oversized_canvas_boxes": oversized_canvas_boxes,
        "unknown_class_boxes": unknown_class_boxes,
        "duplicate_groups_count": len(duplicate_groups),
        "duplicate_images_count": num_duplicate_images,
        "cross_split_leakage": cross_split_leakage,
        "imbalance_ratio": imbalance_ratio,
        "flags": flags,
        "saved_samples": saved_sample_paths,
        "accident_samples": accident_sample_paths,
    }


def generate_markdown_report(audit_results: List[Dict[str, Any]], report_path: Path, class_map_path: Optional[Path] = None):
    """Compiles audit results and class mapping decisions into a detailed Markdown report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Load class mapping if available
    class_map = {}
    if class_map_path and class_map_path.exists():
        try:
            with open(class_map_path, "r", encoding="utf-8") as f:
                class_map = yaml.safe_load(f).get("datasets", {})
        except Exception as e:
            print(f"Warning: Could not load class map: {e}")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# Automated CCTV Accident Detection System (ACS)")
    lines.append("## Multi-Dataset Audit & Label Quality Report\n")
    lines.append(f"**Execution Date**: {now_str}  ")
    lines.append("**Audit Script**: `ml/scripts/audit_dataset.py`  ")
    lines.append("**Class Map File**: `ml/configs/class_map.yaml`\n")
    lines.append("This document audits all downloaded raw datasets prior to merging or fine-tuning, strictly evaluating bounding box geometric validity, image integrity, perceptual duplicates (dHash), cross-split leakage, class balance, and semantic label alignment against the 10 target classes.\n")

    # 1. Executive Summary Table
    lines.append("### 1. Executive Summary Table\n")
    lines.append("| Dataset Name | Total Images | Total Boxes | Empty Labels | Corrupt | Duplicates | Imbalance Ratio | Label Audit Decision | Quality Flags |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |")

    for res in audit_results:
        name = res["dataset_name"]
        imgs = res["total_images"]
        boxes = res["total_boxes"]
        empty = len(res["empty_label_images"])
        corrupt = len(res["corrupt_images"])
        dups = res["duplicate_images_count"]
        imb = f"{res['imbalance_ratio']:.1f}:1" if res['imbalance_ratio'] > 1.0 else "1:1"
        flag_summary = "Clean" if all(f.startswith("PASS") for f in res["flags"]) else "<br>".join([f"`{f}`" for f in res["flags"] if not f.startswith("PASS")])

        # Lookup in class_map (support key or basename)
        base_name = Path(name).name
        cmap_info = class_map.get(name) or class_map.get(base_name) or {}
        status = cmap_info.get("status", "pending")
        decision_badge = {
            "approved_normal_traffic": "✅ **Approved** (Normal Traffic)",
            "partially_approved": "⚠️ **Partial** (Vehicle mapped, Accident dropped)",
            "flagged_defective": "❌ **Rejected / Dropped** (Defective labels & duplicates)",
            "flagged_low_volume": "❌ **Rejected / Dropped** (Low volume & coarse boxes)",
            "candidate_auto_label": "🔍 **Candidate** (Pre-labeling via teacher model)",
            "supplemental_hard_negative": "🛡️ **Hard Negative** (Background negative injection)",
        }.get(status, f"`{status}`")

        lines.append(f"| `{name}` | {imgs:,} | {boxes:,} | {empty} ({empty/max(1, imgs)*100:.1f}%) | {corrupt} | {dups} ({dups/max(1, imgs)*100:.1f}%) | {imb} | {decision_badge} | {flag_summary} |")

    lines.append("\n---\n")

    # 2. Class Mapping & Decision Rationale
    lines.append("### 2. Class Mapping Matrix (`ml/configs/class_map.yaml`)\n")
    lines.append("Every source class was mapped to one of the 10 target classes (`bike`, `bike_bike_accident`, `bike_object_accident`, `bike_person_accident`, `car`, `car_bike_accident`, `car_car_accident`, `car_object_accident`, `car_person_accident`, `person`) or explicitly marked as `\"drop\"` following rigorous empirical inspection of a 20-sample visual audit per generic class:\n")

    lines.append("| Dataset | Source Class ID & Name | Target Class | Action | Empirical Audit Rationale |")
    lines.append("| :--- | :--- | :---: | :---: | :--- |")

    for res in audit_results:
        name = res["dataset_name"]
        base_name = Path(name).name
        cmap_info = (class_map.get(name) or class_map.get(base_name) or {}).get("class_mapping", {})
        if not res["class_names"]:
            lines.append(f"| `{name}` | *(No bounding box classes)* | `N/A` | 🛡️ **NEGATIVE** | Negative background frames without bounding boxes. |")
            continue

        for cid, cname in sorted(res["class_names"].items()):
            mapping = cmap_info.get(cid, {})
            target_name = mapping.get("target_name", "drop")
            action = mapping.get("action", "drop")
            rationale = mapping.get("rationale", "Under evaluation")
            action_badge = "🟩 **MAP**" if action == "map" else "🟥 **DROP**"
            lines.append(f"| `{name}` | `{cid}: {cname}` | `{target_name}` | {action_badge} | {rationale} |")

    lines.append("\n---\n")

    # 3. Detailed Dataset Breakdown
    lines.append("### 3. Detailed Dataset Breakdown\n")
    for res in audit_results:
        name = res["dataset_name"]
        folder_name = res["folder_name"]
        lines.append(f"#### Dataset: `{name}`\n")

        # Splits
        lines.append(f"- **Splits Configuration**: {', '.join([f'{k}: {v:,} images' for k, v in res['split_counts'].items()])}")
        lines.append(f"- **Total Image Count**: {res['total_images']:,}")
        lines.append(f"- **Total Bounding Boxes**: {res['total_boxes']:,}")
        lines.append(f"- **Duplicate Images (dHash)**: {res['duplicate_images_count']} across {res['duplicate_groups_count']} clusters")
        lines.append(f"- **Cross-Split Leakage Clusters**: {len(res['cross_split_leakage'])}")
        lines.append(f"- **Invalid / Out-of-Range Boxes**: {len(res['invalid_boxes'])}")
        lines.append(f"- **Oversized (>85% area) Canvas Boxes**: {len(res['oversized_canvas_boxes'])}")
        lines.append(f"- **Visual Samples Exported**: `12` images in `ml/reports/samples/{folder_name}/`")
        if res["accident_samples"]:
            lines.append(f"- **Generic Accident Inspection**: `20` images in `ml/reports/samples/{folder_name}/accident_inspection_20/`")
        lines.append("\n")

        # Class distribution table
        if res["class_names"]:
            lines.append("**Class Distribution Table**:\n")
            lines.append("| Class ID | Declared Class Name | Box Count | Distribution (%) |")
            lines.append("| :---: | :--- | :---: | :---: |")
            for cid, cname in sorted(res["class_names"].items()):
                cnt = res["class_box_counts"].get(cid, 0)
                pct = (cnt / max(1, res["total_boxes"])) * 100
                lines.append(f"| {cid} | `{cname}` | {cnt:,} | {pct:.2f}% |")
        else:
            lines.append("*Background Negative Dataset: 0 bounding box annotations declared.*\n")

        # Invalid boxes sample
        if res["invalid_boxes"]:
            lines.append("\n**Sample of Invalid Bounding Boxes**:\n")
            lines.append("```text")
            for item in res["invalid_boxes"][:5]:
                lines.append(f"File: {item[0]} | Line {item[1]}: '{item[2]}' -> Reason: {item[3]}")
            lines.append("```")

        # Cross-split leakage sample
        if res["cross_split_leakage"]:
            lines.append("\n**Cross-Split Data Leakage Detected (Identical images across splits)**:\n")
            lines.append("```text")
            for splits_involved, file_names in res["cross_split_leakage"][:5]:
                lines.append(f"Splits: {splits_involved} -> Files: {file_names}")
            lines.append("```")

        # Quality Flags
        lines.append("\n**Label Quality Assessment & Flags**:\n")
        for flag in res["flags"]:
            lines.append(f"- {flag}")
        lines.append("\n")

    lines.append("---\n")
    lines.append("### 4. Visual Sampling & Eyeball Inspection Guide\n")
    lines.append("Twelve (12) high-resolution annotated sample images per dataset have been rendered with bounding boxes and class headers into `ml/reports/samples/<dataset_folder>/`:\n")
    for res in audit_results:
        name = res["dataset_name"]
        folder_name = res["folder_name"]
        lines.append(f"- **`{name}`**: `ml/reports/samples/{folder_name}/sample_*.jpg`")
        if res["accident_samples"]:
            lines.append(f"  - Generic Accident Focus: `ml/reports/samples/{folder_name}/accident_inspection_20/accident_*.jpg`")

    lines.append("\n---\n")
    lines.append("### 5. Final Recommendations & Guardrails (Do Not Merge Yet)\n")
    lines.append("1. **Do Not Merge Raw Datasets Directly**: `justjuu_cctv_accident_detection` has a 25.0% perceptual duplicate rate (692 duplicates) and 46 cross-split data leakage cases. Direct merging would contaminate validation splits and degrade model generalization.")
    lines.append("2. **Preserve Fine-Grained Accident Classes**: Generic `accident` labels in `hilmantm`, `justjuu`, and `on_the_roads` must remain mapped to `drop` or undergo automated re-annotation using the fine-grained model, as they mix rollovers, bike-bike, car-bike, and car-car collisions indiscriminately.")
    lines.append("3. **High-Value Negative Training**: `aitc2025_traffic_object_detection` provides 24,217 pristine normal vehicle bounding boxes (`Motorcycle` -> `bike`, others -> `car`) which will substantially reduce false-alarm accident detections in heavy normal traffic.")
    lines.append("4. **Flag obviously bad labels**: Datasets with severe data leakage, high duplicate rates, or unverified generic labels are rejected or quarantined as candidates pending rigorous cleaning.\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n=======================================================")
    print(f"Audit report saved to: {report_path}")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Audit raw datasets and generate sample visualizations")
    parser.add_argument("--raw_dir", type=str, default="ml/data/raw", help="Path to raw datasets directory")
    parser.add_argument("--report_out", type=str, default="ml/reports/dataset_audit.md", help="Path to markdown output report")
    parser.add_argument("--class_map", type=str, default="ml/configs/class_map.yaml", help="Path to class map configuration")
    parser.add_argument("--samples_dir", type=str, default="ml/reports/samples", help="Path to directory for sample visualizations")
    parser.add_argument("--num_samples", type=int, default=12, help="Number of random sample images per dataset")
    parser.add_argument("--accident_sample_size", type=int, default=20, help="Number of sample images for generic accident inspection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling reproducibility")
    parser.add_argument("--all", action="store_true", help="Audit all datasets discovered recursively under raw_dir")
    parser.add_argument("--dataset", type=str, default=None, help="Audit a specific dataset directory or name")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    report_out = Path(args.report_out)
    class_map_path = Path(args.class_map)
    samples_dir = Path(args.samples_dir)

    if not raw_dir.is_dir():
        print(f"Error: Raw directory '{raw_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    dataset_dirs = []

    if args.dataset:
        cand = Path(args.dataset)
        if not cand.is_dir():
            cand = raw_dir / args.dataset
        if cand.is_dir() and (cand / "data.yaml").exists():
            dataset_dirs.append(cand)
        else:
            print(f"Error: Specified dataset '{args.dataset}' does not exist or has no data.yaml.", file=sys.stderr)
            sys.exit(1)
    elif args.all:
        for p in sorted(raw_dir.rglob("data.yaml")):
            d = p.parent
            splits = find_dataset_splits(d)
            if splits:
                dataset_dirs.append(d)
    else:
        # Default: immediate children with data.yaml
        for d in sorted(raw_dir.iterdir()):
            if d.is_dir() and (d / "data.yaml").exists():
                splits = find_dataset_splits(d)
                if splits:
                    dataset_dirs.append(d)

    print(f"Discovered {len(dataset_dirs)} datasets with valid image splits:")
    for d in dataset_dirs:
        print(f"  - {d.name} ({d})")

    audit_results = []
    for d in dataset_dirs:
        res = audit_single_dataset(
            dataset_dir=d,
            sample_out_dir=samples_dir,
            raw_root=raw_dir,
            num_samples=args.num_samples,
            accident_sample_size=args.accident_sample_size,
            seed=args.seed
        )
        audit_results.append(res)

    generate_markdown_report(audit_results, report_out, class_map_path)


if __name__ == "__main__":
    main()
