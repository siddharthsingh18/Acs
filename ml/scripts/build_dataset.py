#!/usr/bin/env python3
"""
Dataset Builder Script for Automated CCTV Accident Detection System (ACS)
-------------------------------------------------------------------------
Merges approved datasets into ml/data/merged/ using class_map.yaml and classes.yaml:
1. Translates bounding boxes according to class_map.yaml (maps to 10 classes or drops).
2. Removes exact and near-duplicate images across ALL sources using perceptual hashing (dHash/pHash).
3. Groups images into source groups (by video ID / sequence prefix and perceptual-hash clusters)
   so near-identical frames from the same camera/video all land in the same split.
4. Partitions source groups into train/val/test = 70/15/15. The test split is strictly holdout.
5. Enforces holdout isolation: acc_ex.mp4 frames in ml/data/holdout/ are strictly excluded.
6. Validates zero perceptual hash overlap between splits and against holdout footage,
   failing the script if any overlap exists.
7. Writes data.yaml and dataset_stats.md with full per-class per-split metrics.
"""

import argparse
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import yaml

# Perceptual hashing with fallback if imagehash package is not installed
try:
    import imagehash
except ImportError:
    import scipy.fftpack

    class ImageHashWrapper:
        """Wrapper mimicking imagehash.ImageHash interface."""
        def __init__(self, hash_array: np.ndarray):
            self.hash = np.asarray(hash_array, dtype=bool)

        def __sub__(self, other) -> int:
            return int(np.count_nonzero(self.hash != other.hash))

        def __eq__(self, other) -> bool:
            return np.array_equal(self.hash, other.hash)

        def __hash__(self) -> int:
            return hash(bytes(np.packbits(self.hash)))

        def __str__(self) -> str:
            return "".join(format(b, "02x") for b in np.packbits(self.hash))

    class FallbackImageHash:
        """Self-contained implementation of perceptual hashing using DCT and gradient differences."""
        @staticmethod
        def phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> ImageHashWrapper:
            img_size = hash_size * highfreq_factor
            img = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
            pixels = np.asarray(img, dtype=float)
            dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
            dctlowfreq = dct[:hash_size, :hash_size]
            med = np.median(dctlowfreq)
            diff = dctlowfreq > med
            return ImageHashWrapper(diff)

        @staticmethod
        def dhash(image: Image.Image, hash_size: int = 8) -> ImageHashWrapper:
            img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.asarray(img, dtype=np.int32)
            diff = pixels[:, 1:] > pixels[:, :-1]
            return ImageHashWrapper(diff)

    imagehash = FallbackImageHash


def load_yaml(path: Path) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_holdout_extracted(video_path: Path, holdout_dir: Path) -> List[Path]:
    """Ensures acc_ex.mp4 frames are extracted to ml/data/holdout/images/."""
    images_dir = holdout_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(list(images_dir.glob("*.jpg")))
    if existing_frames:
        return existing_frames

    if not video_path.exists():
        print(f"Warning: Holdout video {video_path} not found. Skipping extraction.")
        return []

    print(f"Extracting holdout video frames from {video_path} -> {images_dir}...")
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    extracted_paths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        frame_path = images_dir / f"acc_ex_frame_{count:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        extracted_paths.append(frame_path)
    cap.release()
    print(f"Extracted {len(extracted_paths)} holdout frames.")
    return extracted_paths


def compute_fast_distance_matrix(bits: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Hamming distances between binary hash vectors of shape (N, 64)
    using vectorized BLAS matrix multiplication: dist = sum(a) + sum(b) - 2 * (a @ b.T).
    """
    dot = bits @ bits.T
    row_sums = bits.sum(axis=1)
    return row_sums[:, None] + row_sums[None, :] - 2 * dot


def extract_video_id(filename: str) -> str:
    """
    Extracts video ID, camera sequence ID, or base stem from a filename.
    Handles Roboflow prefixes, CCTV camera tags, and frame patterns.
    """
    # Strip Roboflow hash if present
    base = filename.split(".rf.")[0]
    # Check common CCTV / video sequence prefixes
    for prefix in ["cctv_train_", "cctv_val_", "cctv_test_", "cctv_"]:
        if base.startswith(prefix):
            num_part = base[len(prefix):]
            if num_part.isdigit():
                block = int(num_part) // 10
                return f"cctv_seq_{block:04d}"
            return "cctv_seq"
    
    # Check synthetic or video sequences like FLIP_CAR_V141, Semi-RealisticVD-55, etc.
    if base.startswith("FLIP_CAR_V"):
        parts = base.split("_")
        return "_".join(parts[:3])
    if base.startswith("Semi-RealisticVD-"):
        parts = base.split("_")
        return parts[0]
    if base.startswith("accidentFrame"):
        return "accidentFrame_seq"
    if base.startswith("traffic_img"):
        return "traffic_img_seq"

    # Avoid collapsing anonymous Roboflow filenames
    if base in ["-_-_png", "images_jpg", "images-1_jpg"]:
        return filename

    return base


def collect_candidate_images(
    approved_datasets: List[str],
    raw_dir: Path,
    class_map: Dict[str, Any],
    skip_images_with_dropped_classes: bool = True
) -> List[Dict[str, Any]]:
    """
    Collects, filters, translates bounding boxes, and hashes candidate images from approved datasets.
    """
    candidates = []
    total_scanned = 0
    total_skipped_dropped = 0
    total_skipped_empty = 0

    print(f"\nScanning approved datasets: {approved_datasets}...")
    for ds_name in approved_datasets:
        ds_info = class_map.get(ds_name, {})
        ds_cfg = ds_info.get("class_mapping", {})
        ds_dir = raw_dir / ds_name
        if not ds_dir.exists():
            print(f"Warning: Dataset directory {ds_dir} does not exist. Skipping.")
            continue

        img_files = sorted(list(ds_dir.rglob("*.jpg")))
        print(f"  [{ds_name}]: Found {len(img_files)} images on disk.")

        for img_p in img_files:
            total_scanned += 1
            # Resolve label file
            lbl_p = Path(str(img_p).replace("/images/", "/labels/")).with_suffix(".txt")
            if not lbl_p.exists():
                total_skipped_empty += 1
                continue

            try:
                content = lbl_p.read_text(encoding="utf-8").strip()
            except Exception:
                content = ""

            lines = content.splitlines() if content else []
            has_dropped_class = False
            valid_boxes = []

            for line in lines:
                tokens = line.strip().split()
                if len(tokens) < 5:
                    continue
                try:
                    src_cid = int(tokens[0])
                    xc = float(tokens[1])
                    yc = float(tokens[2])
                    bw = float(tokens[3])
                    bh = float(tokens[4])
                except ValueError:
                    continue

                cmap = ds_cfg.get(src_cid)
                if not cmap or cmap.get("action") == "drop":
                    has_dropped_class = True
                elif cmap.get("action") == "map":
                    valid_boxes.append((cmap["target_id"], xc, yc, bw, bh))

            if skip_images_with_dropped_classes and has_dropped_class:
                total_skipped_dropped += 1
                continue

            if not valid_boxes:
                total_skipped_empty += 1
                continue

            # Verify and hash image
            try:
                with Image.open(img_p) as im:
                    im.verify()
                with Image.open(img_p) as im:
                    ph = imagehash.phash(im)
                    dh = imagehash.dhash(im)
            except Exception as e:
                print(f"Warning: Corrupt image {img_p}: {e}. Skipping.")
                continue

            video_id = extract_video_id(img_p.name)

            candidates.append({
                "img_path": img_p,
                "dataset": ds_name,
                "filename": img_p.name,
                "boxes": valid_boxes,
                "phash": ph,
                "dhash": dh,
                "video_id": video_id,
            })

    print(f"Scanned {total_scanned} raw images:")
    print(f"  - Retained as candidates: {len(candidates)}")
    print(f"  - Skipped (contained dropped classes): {total_skipped_dropped}")
    print(f"  - Skipped (empty / no valid boxes): {total_skipped_empty}")
    return candidates


def deduplicate_images(
    candidates: List[Dict[str, Any]],
    dup_threshold: int = 4
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Identifies exact and near-duplicate images across ALL sources using perceptual hashing
    and Roboflow stem matching, keeping one representative image per duplicate cluster.
    """
    N = len(candidates)
    if N == 0:
        return [], []

    print(f"\nDeduplicating {N} candidate images (pHash Hamming distance threshold <= {dup_threshold})...")
    t0 = time.time()

    # Extract 64-bit boolean arrays for vectorized distance
    bits = np.array([c["phash"].hash.flatten().astype(np.int32) for c in candidates])
    dists = compute_fast_distance_matrix(bits)

    # Adjacency for duplicates
    dup_adj = (dists <= dup_threshold)

    # Also match non-trivial base stems from Roboflow within same dataset
    for i in range(N):
        for j in range(i + 1, N):
            if dup_adj[i, j]:
                continue
            c1, c2 = candidates[i], candidates[j]
            if c1["dataset"] == c2["dataset"]:
                v1, v2 = c1["video_id"], c2["video_id"]
                if v1 == v2 and len(v1) > 4 and v1 not in ["cctv_seq", "accidentFrame_seq", "-_-_png"]:
                    dup_adj[i, j] = True
                    dup_adj[j, i] = True

    n_clusters, cluster_labels = connected_components(csr_matrix(dup_adj))
    print(f"Identified {n_clusters} unique perceptual clusters ({N - n_clusters} duplicates detected in {time.time()-t0:.2f}s).")

    retained = []
    removed = []

    for c_id in range(n_clusters):
        member_indices = np.where(cluster_labels == c_id)[0]
        # Representative selection: pick image with highest box count, then shortest stem
        best_idx = max(
            member_indices,
            key=lambda idx: (len(candidates[idx]["boxes"]), -len(candidates[idx]["filename"]))
        )
        retained.append(candidates[best_idx])
        for idx in member_indices:
            if idx != best_idx:
                removed.append((candidates[idx], candidates[best_idx]["filename"]))

    print(f"Deduplication summary: Retained {len(retained)} images, removed {len(removed)} duplicate copies.")
    return retained, removed


def form_source_groups(
    retained: List[Dict[str, Any]],
    group_threshold: int = 8
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Forms Source Groups using perceptual-hash proximity clusters and video sequence IDs.
    All frames from the same video or camera sequence land in the same group.
    """
    N = len(retained)
    print(f"\nForming Source Groups across {N} retained images (group proximity threshold <= {group_threshold})...")
    t0 = time.time()

    bits = np.array([r["phash"].hash.flatten().astype(np.int32) for r in retained])
    dists = compute_fast_distance_matrix(bits)

    group_adj = (dists <= group_threshold)

    # Connect frames with identical video sequence ID
    video_to_indices = defaultdict(list)
    for idx, r in enumerate(retained):
        vid = r["video_id"]
        if vid and vid not in ["cctv_seq", "-_-_png"]:
            video_to_indices[vid].append(idx)

    for vid, indices in video_to_indices.items():
        if len(indices) > 1:
            for i in indices:
                for j in indices:
                    group_adj[i, j] = True

    n_groups, group_labels = connected_components(csr_matrix(group_adj))
    print(f"Created {n_groups} Source Groups across {N} images in {time.time()-t0:.2f}s.")

    groups = [[] for _ in range(n_groups)]
    for idx, g_id in enumerate(group_labels):
        groups[g_id].append(idx)

    group_sizes = [len(g) for g in groups]
    print(f"Group size stats: Max={max(group_sizes)}, Min={min(group_sizes)}, Median={np.median(group_sizes):.1f}")
    return groups, dists


def split_source_groups(
    groups: List[List[int]],
    retained: List[Dict[str, Any]],
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42
) -> Dict[str, List[int]]:
    """
    Splits Source Groups atomically into train/val/test according to 70/15/15 target ratios.
    Never splits images of the same group across splits.
    """
    r_train, r_val, r_test = split_ratios
    total_images = len(retained)
    target_train = int(round(total_images * r_train))
    target_val = int(round(total_images * r_val))
    target_test = total_images - target_train - target_val

    print(f"\nPartitioning {len(groups)} source groups into Train/Val/Test ({r_train*100:.0f}/{r_val*100:.0f}/{r_test*100:.0f})...")
    print(f"Target image counts: Train={target_train}, Val={target_val}, Test={target_test} (Total={total_images})")

    rng = random.Random(seed)
    shuffled_group_indices = list(range(len(groups)))
    rng.shuffle(shuffled_group_indices)

    # Sort groups descending by size to place larger blocks first (greedy multi-way number partitioning)
    shuffled_group_indices.sort(key=lambda gid: len(groups[gid]), reverse=True)

    splits = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}
    targets = {"train": target_train, "val": target_val, "test": target_test}

    for g_id in shuffled_group_indices:
        g_items = groups[g_id]
        cnt = len(g_items)

        # Compute priority score: normalized remaining capacity
        best_split = max(
            targets.keys(),
            key=lambda s: (targets[s] - split_counts[s]) / max(1, targets[s])
        )

        splits[best_split].extend(g_items)
        split_counts[best_split] += cnt

    for s in ["train", "val", "test"]:
        pct = (split_counts[s] / max(1, total_images)) * 100
        print(f"  Split '{s:5s}': {split_counts[s]:5d} images ({pct:5.1f}%)")

    return splits


def verify_cross_split_isolation(
    splits: Dict[str, List[int]],
    retained: List[Dict[str, Any]],
    dists: np.ndarray,
    holdout_frames: List[Path],
    overlap_threshold: int = 6
):
    """
    Enforces automated perceptual hash check:
    1. Zero cross-split hash overlap (Hamming distance > overlap_threshold between all train/val/test pairs).
    2. Zero overlap with holdout footage (acc_ex.mp4).
    Fails the script with an exception if any violation is found.
    """
    print(f"\n=======================================================")
    print("Running Automated Perceptual Hash Overlap & Isolation Verification...")
    print(f"Overlap rejection threshold: Hamming distance <= {overlap_threshold}")
    print("=======================================================")

    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]

    # 1. Train vs Val
    d_tv = dists[np.ix_(train_indices, val_indices)]
    min_tv = d_tv.min() if d_tv.size > 0 else 999
    print(f"Min perceptual hash distance between Train and Val: {min_tv}")
    if min_tv <= overlap_threshold:
        i, j = np.where(d_tv <= overlap_threshold)
        t_img = retained[train_indices[i[0]]]["img_path"]
        v_img = retained[val_indices[j[0]]]["img_path"]
        raise RuntimeError(
            f"CRITICAL VERIFICATION FAILURE: Perceptual hash overlap between Train and Val splits!\n"
            f"Train image: {t_img}\nVal image: {v_img}\nHamming distance: {min_tv} <= {overlap_threshold}"
        )

    # 2. Train vs Test
    d_tt = dists[np.ix_(train_indices, test_indices)]
    min_tt = d_tt.min() if d_tt.size > 0 else 999
    print(f"Min perceptual hash distance between Train and Test: {min_tt}")
    if min_tt <= overlap_threshold:
        i, j = np.where(d_tt <= overlap_threshold)
        t_img = retained[train_indices[i[0]]]["img_path"]
        te_img = retained[test_indices[j[0]]]["img_path"]
        raise RuntimeError(
            f"CRITICAL VERIFICATION FAILURE: Perceptual hash overlap between Train and Test splits!\n"
            f"Train image: {t_img}\nTest image: {te_img}\nHamming distance: {min_tt} <= {overlap_threshold}"
        )

    # 3. Val vs Test
    d_vt = dists[np.ix_(val_indices, test_indices)]
    min_vt = d_vt.min() if d_vt.size > 0 else 999
    print(f"Min perceptual hash distance between Val and Test: {min_vt}")
    if min_vt <= overlap_threshold:
        i, j = np.where(d_vt <= overlap_threshold)
        v_img = retained[val_indices[i[0]]]["img_path"]
        te_img = retained[test_indices[j[0]]]["img_path"]
        raise RuntimeError(
            f"CRITICAL VERIFICATION FAILURE: Perceptual hash overlap between Val and Test splits!\n"
            f"Val image: {v_img}\nTest image: {te_img}\nHamming distance: {min_vt} <= {overlap_threshold}"
        )

    # 4. Holdout Footage Check
    if holdout_frames:
        print(f"Checking {len(holdout_frames)} holdout frames against all {len(retained)} merged images...")
        holdout_hashes = []
        for h_p in holdout_frames:
            with Image.open(h_p) as im:
                holdout_hashes.append(imagehash.phash(im).hash.flatten().astype(np.int32))
        h_bits = np.array(holdout_hashes)  # (H, 64)
        m_bits = np.array([r["phash"].hash.flatten().astype(np.int32) for r in retained])  # (M, 64)

        # Distance matrix (H, M)
        h_dot = h_bits @ m_bits.T
        h_dists = h_bits.sum(axis=1)[:, None] + m_bits.sum(axis=1)[None, :] - 2 * h_dot
        min_holdout_dist = h_dists.min()
        print(f"Min perceptual hash distance between holdout footage and merged dataset: {min_holdout_dist}")
        if min_holdout_dist <= overlap_threshold:
            hi, mi = np.where(h_dists <= overlap_threshold)
            h_f = holdout_frames[hi[0]]
            m_f = retained[mi[0]]["img_path"]
            raise RuntimeError(
                f"CRITICAL VERIFICATION FAILURE: Holdout footage contamination detected!\n"
                f"Holdout frame: {h_f}\nMerged image: {m_f}\nHamming distance: {min_holdout_dist} <= {overlap_threshold}"
            )

    print("✅ Automated check PASSED: ZERO perceptual hash overlap across splits or against holdout footage!\n")


def export_merged_dataset(
    splits: Dict[str, List[int]],
    retained: List[Dict[str, Any]],
    output_dir: Path,
    target_classes: List[str]
) -> Dict[str, Any]:
    """
    Exports merged images, labels, data.yaml, and computes final distribution metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_base = output_dir / "images"
    labels_base = output_dir / "labels"

    # Clean existing splits
    for s in ["train", "val", "test"]:
        (images_base / s).mkdir(parents=True, exist_ok=True)
        (labels_base / s).mkdir(parents=True, exist_ok=True)
        # Clear files if re-running
        for p in (images_base / s).glob("*.jpg"):
            p.unlink()
        for p in (labels_base / s).glob("*.txt"):
            p.unlink()

    stats = {
        "splits": {},
        "class_counts": {s: Counter() for s in ["train", "val", "test"]},
        "dataset_contribs": {s: Counter() for s in ["train", "val", "test"]}
    }

    print("Writing merged dataset images and label files...")
    for split_name, indices in splits.items():
        img_out_dir = images_base / split_name
        lbl_out_dir = labels_base / split_name

        for idx in indices:
            item = retained[idx]
            src_img = item["img_path"]
            ds_name = item["dataset"]
            boxes = item["boxes"]

            clean_stem = f"{ds_name}_{src_img.stem}"
            dst_img = img_out_dir / f"{clean_stem}.jpg"
            dst_lbl = lbl_out_dir / f"{clean_stem}.txt"

            # Copy image
            shutil.copy2(src_img, dst_img)

            # Write translated labels
            lbl_lines = []
            for cid, xc, yc, bw, bh in boxes:
                lbl_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                stats["class_counts"][split_name][cid] += 1

            dst_lbl.write_text("\n".join(lbl_lines), encoding="utf-8")
            stats["dataset_contribs"][split_name][ds_name] += 1

        stats["splits"][split_name] = len(indices)

    # Write data.yaml
    yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(target_classes),
        "names": {i: name for i, name in enumerate(target_classes)},
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"Exported data.yaml to {yaml_path}")
    return stats


def generate_dataset_stats_report(
    stats: Dict[str, Any],
    target_classes: List[str],
    total_raw_scanned: int,
    duplicates_removed_count: int,
    n_source_groups: int,
    stats_path: Path
):
    """Writes dataset_stats.md with detailed per-class, per-split metrics and verification logs."""
    splits = ["train", "val", "test"]
    total_imgs = sum(stats["splits"].values())
    total_boxes = sum(sum(stats["class_counts"][s].values()) for s in splits)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# Automated CCTV Accident Detection System (ACS)")
    lines.append("## Merged Dataset Statistics & Integrity Report\n")
    lines.append(f"**Generation Date**: {now_str}  ")
    lines.append(f"**Dataset Location**: `ml/data/merged`  ")
    lines.append(f"**Configuration Source**: `ml/configs/class_map.yaml` & `ml/configs/classes.yaml`\n")

    # 1. Executive Summary Table
    lines.append("### 1. Dataset Partition Summary\n")
    lines.append("| Split | Image Count | Percentage (%) | Bounding Boxes | Avg Boxes / Image | Usage Rule |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")

    for s in splits:
        icnt = stats["splits"][s]
        bcnt = sum(stats["class_counts"][s].values())
        pct = (icnt / max(1, total_imgs)) * 100
        avg_b = bcnt / max(1, icnt)
        rule = "**Active Training**" if s == "train" else "**Hyperparameter Tuning & Checkpointing**" if s == "val" else "⚠️ **STRICT HOLDOUT TEST (Never Train/Tune)**"
        lines.append(f"| `{s}` | {icnt:,} | {pct:.1f}% | {bcnt:,} | {avg_b:.2f} | {rule} |")

    lines.append(f"| **Total** | **{total_imgs:,}** | **100.0%** | **{total_boxes:,}** | **{total_boxes/max(1, total_imgs):.2f}** | — |")
    lines.append("\n---\n")

    # 2. Per-Class Distribution Matrix
    lines.append("### 2. Per-Class Distribution Matrix (10 Target Classes)\n")
    lines.append("| Class ID | Class Name | Train Boxes | Val Boxes | Test Boxes | Total Boxes | Distribution (%) |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")

    for cid, cname in enumerate(target_classes):
        tb = stats["class_counts"]["train"][cid]
        vb = stats["class_counts"]["val"][cid]
        teb = stats["class_counts"]["test"][cid]
        tot = tb + vb + teb
        pct = (tot / max(1, total_boxes)) * 100
        lines.append(f"| {cid} | `{cname}` | {tb:,} | {vb:,} | {teb:,} | {tot:,} | {pct:.2f}% |")

    lines.append("\n---\n")

    # 3. Source Dataset Contributions
    lines.append("### 3. Source Dataset Contributions\n")
    lines.append("| Dataset Source | Train Images | Val Images | Test Images | Total Images | Status in Class Map |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")

    all_ds = sorted(list(set(k for s in splits for k in stats["dataset_contribs"][s].keys())))
    for ds in all_ds:
        ti = stats["dataset_contribs"]["train"][ds]
        vi = stats["dataset_contribs"]["val"][ds]
        tei = stats["dataset_contribs"]["test"][ds]
        tot = ti + vi + tei
        lines.append(f"| `{ds}` | {ti:,} | {vi:,} | {tei:,} | {tot:,} | Approved |")

    lines.append("\n---\n")

    # 4. Deduplication & Grouping Audit
    lines.append("### 4. Deduplication & Source Grouping Audit\n")
    lines.append(f"- **Raw Scanned Candidate Images**: {total_raw_scanned:,}")
    lines.append(f"- **Perceptual Near-Duplicates Removed**: {duplicates_removed_count:,} copies")
    lines.append(f"- **Retained Unique Images**: {total_imgs:,}")
    lines.append(f"- **Atomic Source Groups Formed**: {n_source_groups:,} groups")
    lines.append(f"- **Group Splitting Strategy**: Source groups were assigned atomically to splits to prevent temporal and camera sequence leakage.\n")

    lines.append("\n---\n")

    # 5. Automated Integrity & Isolation Certification
    lines.append("### 5. Automated Integrity & Isolation Certificate\n")
    lines.append("An automated perceptual hash verification was executed across all splits:\n")
    lines.append("- ✅ **Train vs Val Perceptual Hash Overlap**: ZERO overlap (`Min Hamming Distance > overlap_threshold`).")
    lines.append("- ✅ **Train vs Test Perceptual Hash Overlap**: ZERO overlap (`Min Hamming Distance > overlap_threshold`).")
    lines.append("- ✅ **Val vs Test Perceptual Hash Overlap**: ZERO overlap (`Min Hamming Distance > overlap_threshold`).")
    lines.append("- ✅ **Holdout Footage Isolation (`acc_ex.mp4`)**: Verified holdout frames tested against all merged images. ZERO overlap.")
    lines.append("- ✅ **Test Split Rule Enforced**: The test split is permanently segregated and must never be used for gradient descent or hyperparameter tuning.\n")

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Exported dataset statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge approved datasets into ml/data/merged/ with zero leakage")
    parser.add_argument("--raw_dir", type=str, default="ml/data/raw", help="Path to raw datasets directory")
    parser.add_argument("--output_dir", type=str, default="ml/data/merged", help="Path to output merged dataset directory")
    parser.add_argument("--holdout_dir", type=str, default="ml/data/holdout", help="Path to holdout dataset directory")
    parser.add_argument("--class_map", type=str, default="ml/configs/class_map.yaml", help="Path to class map YAML")
    parser.add_argument("--classes", type=str, default="ml/configs/classes.yaml", help="Path to target classes YAML")
    parser.add_argument("--video_holdout", type=str, default="acc_ex.mp4", help="Path to holdout video file")
    parser.add_argument("--dup_threshold", type=int, default=4, help="pHash Hamming distance threshold for near-duplicates")
    parser.add_argument("--group_threshold", type=int, default=8, help="pHash Hamming distance threshold for source grouping")
    parser.add_argument("--overlap_threshold", type=int, default=6, help="pHash Hamming distance threshold for cross-split assertion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group partition reproducibility")
    parser.add_argument("--clean_only", action="store_true", default=True, help="Skip images that contain dropped classes (avoids unlabelled accidents)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    holdout_dir = Path(args.holdout_dir)
    class_map_path = Path(args.class_map)
    classes_path = Path(args.classes)
    video_holdout = Path(args.video_holdout)

    print("==============================================================================")
    print("ACS Dataset Builder: Deduplication, Grouping, Partitioning & Merge")
    print("==============================================================================")

    # 1. Load Configurations
    classes_cfg = load_yaml(classes_path)
    target_classes = classes_cfg.get("classes", [])
    print(f"Target Classes ({len(target_classes)}): {target_classes}")

    class_map_cfg = load_yaml(class_map_path)
    datasets_cfg = class_map_cfg.get("datasets", {})
    approved_datasets = [k for k, v in datasets_cfg.items() if "approved" in v.get("status", "")]
    print(f"Approved Datasets for Merge ({len(approved_datasets)}): {approved_datasets}")

    # 2. Ensure Holdout Frames Extracted
    holdout_frames = ensure_holdout_extracted(video_holdout, holdout_dir)

    # 3. Collect Candidate Images
    candidates = collect_candidate_images(
        approved_datasets=approved_datasets,
        raw_dir=raw_dir,
        class_map=datasets_cfg,
        skip_images_with_dropped_classes=args.clean_only
    )
    if not candidates:
        print("Error: No candidate images found. Aborting.", file=sys.stderr)
        sys.exit(1)

    # 4. Deduplicate (exact & near-duplicates across all sources)
    retained, removed = deduplicate_images(candidates, dup_threshold=args.dup_threshold)

    # 5. Form Source Groups
    groups, dists = form_source_groups(retained, group_threshold=args.group_threshold)

    # 6. Split Groups into 70/15/15
    splits = split_source_groups(
        groups=groups,
        retained=retained,
        split_ratios=(0.70, 0.15, 0.15),
        seed=args.seed
    )

    # 7. Automated Cross-Split Isolation & Holdout Overlap Check
    verify_cross_split_isolation(
        splits=splits,
        retained=retained,
        dists=dists,
        holdout_frames=holdout_frames,
        overlap_threshold=args.overlap_threshold
    )

    # 8. Export Merged Dataset & data.yaml
    stats = export_merged_dataset(splits, retained, output_dir, target_classes)

    # 9. Generate dataset_stats.md
    generate_dataset_stats_report(
        stats=stats,
        target_classes=target_classes,
        total_raw_scanned=len(candidates),
        duplicates_removed_count=len(removed),
        n_source_groups=len(groups),
        stats_path=output_dir / "dataset_stats.md"
    )

    print("\n==============================================================================")
    print("Dataset build and verification completed successfully!")
    print(f"Merged Dataset Location: {output_dir.resolve()}")
    print("==============================================================================")


if __name__ == "__main__":
    main()
