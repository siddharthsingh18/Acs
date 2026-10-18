#!/usr/bin/env python3
"""
Automated CCTV Accident Detection System (ACS)
Monthly Retraining & Continuous Learning Workflow (retrain_workflow.py)

Workflow:
1. Merge Staging: Ingests reviewed feedback from ml/data/feedback_staging/ (hard negatives & candidates).
2. Perceptual Deduplication & Group-Split: Runs dHash/pHash dedup, groups by source clip/sequence,
   and splits into candidate train & val splits.
3. Frozen Test Set Protection: Strictly enforces that ml/data/merged/images/test and holdout video frames
   are NEVER in the training or val splits. Asserts zero perceptual overlap (Hamming distance > 6).
4. Train: Fine-tunes YOLO model on the updated training pool with deterministic seeding.
5. Evaluate against Frozen Test Set: Computes accident-class macro mAP50 with bootstrap CIs and false alarm rates.
6. Promotion Rule: Promotes candidate checkpoint to production ONLY IF it satisfies all 3 criteria:
   (a) Beats baseline on accident-class macro mAP50 beyond bootstrap CI (diff CI lower bound > 0.0),
   (b) Has no worse false alarms per hour,
   (c) Runs at >= 15 FPS on CPU.
"""

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import yaml
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acs.config import ACCIDENT_CLASSES, CLASS_NAMES, config
import imagehash
from ml.scripts.promote import (
    benchmark_inference_speed,
    evaluate_candidate,
    evaluate_test_clips_telemetry,
    execute_promotion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("acs.ml.retrain_workflow")

DEFAULT_STAGING_DIR = PROJECT_ROOT / "ml" / "data" / "feedback_staging"
DEFAULT_BASE_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "merged"
DEFAULT_RETRAIN_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "retrain_merged"
DEFAULT_TEST_VIDEO_DIR = PROJECT_ROOT / "ml" / "data" / "video_eval"
DEFAULT_HOLDOUT_DIR = PROJECT_ROOT / "ml" / "data" / "holdout"


def compute_fast_hamming_matrix(a_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    """Computes pairwise Hamming distance between boolean/int arrays of shape (N, 64) and (M, 64)."""
    dot = a_bits.astype(np.int32) @ b_bits.astype(np.int32).T
    dists = a_bits.sum(axis=1)[:, None] + b_bits.sum(axis=1)[None, :] - 2 * dot
    return dists


def extract_phash_bits(img_path: Path) -> np.ndarray:
    """Computes a 64-bit perceptual hash array for an image."""
    with Image.open(img_path) as im:
        h = imagehash.phash(im)
        return h.hash.flatten().astype(np.int32)


def assert_frozen_test_set_isolation(
    train_images: List[Path],
    val_images: List[Path],
    frozen_test_images: List[Path],
    holdout_images: List[Path],
    overlap_threshold: int = 6,
) -> None:
    """Strictly verifies that NO training or validation image matches any image in the frozen test set

    or holdout benchmark videos. Raises AssertionError if any perceptual leakage is found.
    """
    logger.info("Enforcing FROZEN TEST SET ISOLATION & ZERO LEAKAGE check...")
    logger.info(
        "Candidate Pool: %d train, %d val | Frozen Benchmark: %d test, %d holdout frames",
        len(train_images),
        len(val_images),
        len(frozen_test_images),
        len(holdout_images),
    )

    if not frozen_test_images:
        logger.warning("No frozen test images found to check against.")
        return

    # 1. Compute test set hashes
    test_hashes = np.array([extract_phash_bits(p) for p in frozen_test_images])

    # 2. Compute holdout video hashes if present
    holdout_hashes = (
        np.array([extract_phash_bits(p) for p in holdout_images])
        if holdout_images
        else np.empty((0, 64), dtype=np.int32)
    )

    all_frozen = (
        np.vstack([test_hashes, holdout_hashes])
        if len(holdout_hashes) > 0
        else test_hashes
    )
    frozen_labels = [f"test:{p.name}" for p in frozen_test_images] + [
        f"holdout:{p.name}" for p in holdout_images
    ]

    # 3. Check train pool against frozen benchmark
    if train_images:
        train_hashes = np.array([extract_phash_bits(p) for p in train_images])
        dists_train = compute_fast_hamming_matrix(train_hashes, all_frozen)
        min_train_dist = dists_train.min()
        logger.info("Min Hamming distance between Train pool and Frozen Test/Holdout: %d", min_train_dist)
        if min_train_dist <= overlap_threshold:
            ti, fi = np.where(dists_train <= overlap_threshold)
            t_bad = train_images[ti[0]]
            f_bad = frozen_labels[fi[0]]
            raise AssertionError(
                f"🚨 CRITICAL DATA LEAKAGE: Training pool contains image perceptually identical to frozen test set!\n"
                f"Train Image: {t_bad}\n"
                f"Frozen Reference: {f_bad}\n"
                f"Hamming Distance: {min_train_dist} <= threshold {overlap_threshold}\n"
                f"The frozen test set must NEVER be trained on."
            )

    # 4. Check val pool against frozen benchmark
    if val_images:
        val_hashes = np.array([extract_phash_bits(p) for p in val_images])
        dists_val = compute_fast_hamming_matrix(val_hashes, all_frozen)
        min_val_dist = dists_val.min()
        logger.info("Min Hamming distance between Val pool and Frozen Test/Holdout: %d", min_val_dist)
        if min_val_dist <= overlap_threshold:
            vi, fi = np.where(dists_val <= overlap_threshold)
            v_bad = val_images[vi[0]]
            f_bad = frozen_labels[fi[0]]
            raise AssertionError(
                f"🚨 CRITICAL DATA LEAKAGE: Validation pool contains image perceptually identical to frozen test set!\n"
                f"Val Image: {v_bad}\n"
                f"Frozen Reference: {f_bad}\n"
                f"Hamming Distance: {min_val_dist} <= threshold {overlap_threshold}"
            )

    logger.info("✅ FROZEN TEST SET ISOLATION PASSED: Zero overlap detected across splits!")


def merge_staging_and_build_retrain_splits(
    base_data_dir: Path,
    staging_dir: Path,
    output_dir: Path,
    holdout_dir: Path,
    val_fraction: float = 0.15,
) -> Dict[str, Any]:
    """Merges staging data with baseline train/val data, runs perceptual deduplication,

    performs group-splitting by source clip, enforces frozen test set isolation,
    and writes out retrain_merged/ dataset.
    """
    logger.info("Merging staging dataset from %s into %s...", staging_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_images_train = output_dir / "images" / "train"
    out_images_val = output_dir / "images" / "val"
    out_images_test = output_dir / "images" / "test"
    out_labels_train = output_dir / "labels" / "train"
    out_labels_val = output_dir / "labels" / "val"
    out_labels_test = output_dir / "labels" / "test"

    for d in [
        out_images_train,
        out_images_val,
        out_images_test,
        out_labels_train,
        out_labels_val,
        out_labels_test,
    ]:
        d.mkdir(parents=True, exist_ok=True)
        # Clear existing
        for f in d.glob("*.*"):
            try:
                f.unlink()
            except Exception:
                pass

    # 1. Strictly copy FROZEN test set as-is from baseline
    src_test_images_dir = base_data_dir / "images" / "test"
    src_test_labels_dir = base_data_dir / "labels" / "test"
    frozen_test_images: List[Path] = []
    if src_test_images_dir.exists():
        for img in src_test_images_dir.glob("*.jpg"):
            dst_img = out_images_test / img.name
            shutil.copy2(img, dst_img)
            frozen_test_images.append(dst_img)
            lbl = src_test_labels_dir / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, out_labels_test / lbl.name)

    logger.info("Copied %d strictly frozen test images.", len(frozen_test_images))

    # 2. Gather candidate images from baseline train & val splits
    candidate_items: List[Dict[str, Any]] = []
    for split in ["train", "val"]:
        img_dir = base_data_dir / "images" / split
        lbl_dir = base_data_dir / "labels" / split
        if img_dir.exists():
            for img in img_dir.glob("*.jpg"):
                lbl = lbl_dir / f"{img.stem}.txt"
                prefix = img.stem.split("_")[0] if "_" in img.stem else img.stem
                candidate_items.append({
                    "source": "baseline",
                    "group": f"base_{prefix}",
                    "img_path": img,
                    "lbl_path": lbl if lbl.exists() else None,
                })

    # 3. Gather staging images from feedback_staging
    staging_images_dir = staging_dir / "images"
    staging_labels_dir = staging_dir / "labels"
    staged_count = 0
    if staging_images_dir.exists():
        for img in staging_images_dir.glob("*.jpg"):
            lbl = staging_labels_dir / f"{img.stem}.txt"
            prefix = img.stem.split("_")[1] if "_" in img.stem else img.stem
            candidate_items.append({
                "source": "staging",
                "group": f"staging_{prefix}",
                "img_path": img,
                "lbl_path": lbl if lbl.exists() else None,
            })
            staged_count += 1

    logger.info("Total candidate pool: %d images (%d from staging).", len(candidate_items), staged_count)

    # 4. Perceptual deduplication within candidate pool
    logger.info("Running perceptual hash deduplication on candidate pool...")
    unique_items: List[Dict[str, Any]] = []
    seen_hashes: List[np.ndarray] = []

    for item in candidate_items:
        try:
            h = extract_phash_bits(item["img_path"])
        except Exception as e:
            logger.warning("Could not compute hash for %s: %s", item["img_path"], e)
            continue

        is_dup = False
        if seen_hashes:
            h_matrix = np.array(seen_hashes)
            dists = compute_fast_hamming_matrix(h[None, :], h_matrix)
            if dists.min() <= 3:  # Near-duplicate threshold
                is_dup = True

        if not is_dup:
            seen_hashes.append(h)
            unique_items.append(item)

    logger.info("Deduplication kept %d unique candidate images (pruned %d duplicates).",
                len(unique_items), len(candidate_items) - len(unique_items))

    # 5. Group-split partitioning (by group/sequence so clip frames don't cross splits)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in unique_items:
        grp = item["group"]
        groups.setdefault(grp, []).append(item)

    sorted_groups = sorted(groups.items(), key=lambda g: len(g[1]), reverse=True)
    train_items: List[Dict[str, Any]] = []
    val_items: List[Dict[str, Any]] = []

    total_unique = len(unique_items)
    val_target = int(total_unique * val_fraction)
    val_count = 0

    for grp_name, items in sorted_groups:
        if val_count < val_target:
            val_items.extend(items)
            val_count += len(items)
        else:
            train_items.extend(items)

    logger.info("Partitioned candidate groups: %d train images, %d val images.", len(train_items), len(val_items))

    # 6. Copy train and val items to retrain directory
    train_img_paths: List[Path] = []
    for item in train_items:
        dst_img = out_images_train / item["img_path"].name
        shutil.copy2(item["img_path"], dst_img)
        train_img_paths.append(dst_img)
        if item["lbl_path"] and item["lbl_path"].exists():
            shutil.copy2(item["lbl_path"], out_labels_train / item["lbl_path"].name)
        else:
            # Empty label file for hard negative
            with open(out_labels_train / f"{dst_img.stem}.txt", "w") as f:
                pass

    val_img_paths: List[Path] = []
    for item in val_items:
        dst_img = out_images_val / item["img_path"].name
        shutil.copy2(item["img_path"], dst_img)
        val_img_paths.append(dst_img)
        if item["lbl_path"] and item["lbl_path"].exists():
            shutil.copy2(item["lbl_path"], out_labels_val / item["lbl_path"].name)
        else:
            with open(out_labels_val / f"{dst_img.stem}.txt", "w") as f:
                pass

    # 7. Collect holdout frames
    holdout_images_dir = holdout_dir / "images"
    holdout_images = (
        sorted(list(holdout_images_dir.glob("*.jpg")))
        if holdout_images_dir.exists()
        else []
    )

    # 8. Assert strict frozen test set isolation
    assert_frozen_test_set_isolation(
        train_images=train_img_paths,
        val_images=val_img_paths,
        frozen_test_images=frozen_test_images,
        holdout_images=holdout_images,
    )

    # 9. Write data.yaml for retraining
    retrain_yaml_path = output_dir / "data.yaml"
    data_yaml_doc = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 10,
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(retrain_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_doc, f, sort_keys=False)

    return {
        "output_dir": str(output_dir),
        "data_yaml": str(retrain_yaml_path),
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_test": len(frozen_test_images),
        "staged_count": staged_count,
    }


def run_monthly_retrain_pipeline(
    staging_dir: Path = DEFAULT_STAGING_DIR,
    base_data_dir: Path = DEFAULT_BASE_DATA_DIR,
    output_dir: Path = DEFAULT_RETRAIN_DATA_DIR,
    holdout_dir: Path = DEFAULT_HOLDOUT_DIR,
    test_video_dir: Path = DEFAULT_TEST_VIDEO_DIR,
    epochs: int = 5,
    imgsz: int = 640,
    device: str = "auto",
    skip_train: bool = False,
    force_promote: bool = False,
) -> Dict[str, Any]:
    """Executes the full monthly retraining workflow end-to-end."""
    logger.info("=" * 80)
    logger.info("STARTING MONTHLY RETRAIN WORKFLOW")
    logger.info("=" * 80)

    # Step 1 & 2: Merge staging, dedup, group-split, and verify frozen test isolation
    prep_stats = merge_staging_and_build_retrain_splits(
        base_data_dir=base_data_dir,
        staging_dir=staging_dir,
        output_dir=output_dir,
        holdout_dir=holdout_dir,
    )

    data_yaml_path = Path(prep_stats["data_yaml"])
    baseline_model_path = PROJECT_ROOT / "best.pt"
    if not baseline_model_path.exists():
        baseline_model_path = PROJECT_ROOT / "models" / "accident_v1.pt"
    if not baseline_model_path.exists():
        baseline_model_path = PROJECT_ROOT / "ml" / "baseline" / "yolov8n.pt"

    candidate_checkpoint_path = PROJECT_ROOT / "ml" / "runs" / "retrain_candidate" / "weights" / "best.pt"

    if skip_train:
        logger.info("[Retrain Workflow] --skip-train specified. Using baseline model as candidate for simulation.")
        candidate_checkpoint_path = baseline_model_path
    else:
        logger.info("Training candidate model on updated training pool...")
        resolved_device = device
        if resolved_device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"

        # Initialize from baseline model
        model = YOLO(str(baseline_model_path))
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            device=resolved_device,
            seed=42,
            deterministic=True,
            project=str(PROJECT_ROOT / "ml" / "runs"),
            name="retrain_candidate",
            exist_ok=True,
            verbose=False,
        )
        if not candidate_checkpoint_path.exists():
            # If best.pt not found, check last.pt
            last_pt = PROJECT_ROOT / "ml" / "runs" / "retrain_candidate" / "weights" / "last.pt"
            if last_pt.exists():
                candidate_checkpoint_path = last_pt

    # Step 4 & 5: Evaluate against frozen test set and apply promotion rule
    logger.info("Evaluating candidate checkpoint %s against frozen test set...", candidate_checkpoint_path)
    eval_result = evaluate_candidate(
        candidate_path=candidate_checkpoint_path,
        baseline_path=baseline_model_path,
        data_yaml_path=data_yaml_path,
        test_video_dir=test_video_dir,
        bootstrap_iterations=50 if skip_train else 100,
    )

    promoted = eval_result["promoted"] or force_promote
    if force_promote and not eval_result["promoted"]:
        logger.warning("Promotion rule criteria failed, but --force-promote was requested.")

    if promoted and not skip_train:
        logger.info("Candidate model PASSED all promotion criteria! Promoting to production.")
        execute_promotion(candidate_checkpoint_path)
    else:
        logger.info("Candidate did not satisfy promotion rule. Production baseline retained.")

    # Save summary report
    reports_dir = PROJECT_ROOT / "ml" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"retrain_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = reports_dir / report_filename

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": prep_stats,
        "baseline_model": str(baseline_model_path),
        "candidate_model": str(candidate_checkpoint_path),
        "evaluation": eval_result,
        "promoted": promoted,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Retrain workflow completed. Summary saved to %s", report_path)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="ACS Monthly Retraining & Continuous Learning Workflow"
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        help="Path to operator feedback staging dataset directory.",
    )
    parser.add_argument(
        "--base-data",
        type=Path,
        default=DEFAULT_BASE_DATA_DIR,
        help="Path to baseline merged dataset containing frozen test split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RETRAIN_DATA_DIR,
        help="Path to output retrain dataset directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Model image resolution.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Hardware device ('auto', 'cpu', 'cuda', 'mps').",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the actual training step (validates dataset merge, dedup, and isolation rules).",
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Force promotion of candidate model even if promotion criteria are not met.",
    )
    args = parser.parse_args()

    result = run_monthly_retrain_pipeline(
        staging_dir=args.staging_dir,
        base_data_dir=args.base_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        skip_train=args.skip_train,
        force_promote=args.force_promote,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
