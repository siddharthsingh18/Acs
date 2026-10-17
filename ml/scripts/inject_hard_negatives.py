#!/usr/bin/env python3
"""
Hard Negative Injection Script for ACS Training Set
---------------------------------------------------
Adds background images with empty label files (0 bytes) to ml/data/merged/images/train/
covering all requested negative scenarios:
- normal traffic / intersections (cctv_non_accident)
- traffic jams / dense congestion / braking (traffic_jam)
- parked cars (parked_cars)
- night traffic / dark lighting (traffic_night)
- rain / wet road glare / puddles / reflections (rain_traffic)

Enforces strict zero-leakage verification against ml/data/holdout/ and val/test splits.
"""

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import imagehash
import numpy as np
from PIL import Image


def load_hashes(image_paths: List[Path]) -> Dict[Path, imagehash.ImageHash]:
    """Computes pHash for a list of images."""
    hashes = {}
    for p in image_paths:
        try:
            with Image.open(p) as im:
                hashes[p] = imagehash.phash(im)
        except Exception as e:
            print(f"Warning: could not hash {p}: {e}")
    return hashes


def inject_hard_negatives(
    raw_neg_dir: Path,
    merged_dir: Path,
    holdout_dir: Path,
    target_counts: Dict[str, int],
    min_dist_threshold: int = 8,
    seed: int = 42
) -> Dict[str, Any]:
    """Selects and injects verified hard negatives into ml/data/merged/images/train/."""
    rng = random.Random(seed)
    train_img_dir = merged_dir / "images" / "train"
    train_lbl_dir = merged_dir / "labels" / "train"
    val_img_dir = merged_dir / "images" / "val"
    test_img_dir = merged_dir / "images" / "test"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect protected hashes (Holdout, Val, Test)
    print("Collecting protected hashes (Holdout, Val, Test splits)...")
    holdout_imgs = sorted(list((holdout_dir / "images").glob("*.jpg")))
    val_imgs = sorted(list(val_img_dir.glob("*.jpg")))
    test_imgs = sorted(list(test_img_dir.glob("*.jpg")))

    protected_imgs = holdout_imgs + val_imgs + test_imgs
    print(f"Hashing {len(protected_imgs)} protected images...")
    protected_hashes = []
    for p in protected_imgs:
        with Image.open(p) as im:
            protected_hashes.append(imagehash.phash(im).hash.flatten().astype(np.int32))
    protected_bits = np.array(protected_hashes)  # (P, 64)

    # 2. Sample and verify negatives per domain
    selected_negatives = []
    category_summary = {}

    existing_neg_hashes = []

    print("\nSelecting and validating hard negative images across domains:")
    for domain, count in target_counts.items():
        domain_dir = raw_neg_dir / domain
        if not domain_dir.exists():
            print(f"Warning: Domain directory {domain_dir} does not exist. Skipping.")
            continue

        cand_files = sorted(list(domain_dir.rglob("*.jpg")))
        rng.shuffle(cand_files)
        print(f"  [{domain}]: Found {len(cand_files)} candidate images. Target = {count}")

        accepted_for_domain = []
        for img_p in cand_files:
            if len(accepted_for_domain) >= count:
                break

            try:
                with Image.open(img_p) as im:
                    im.verify()
                with Image.open(img_p) as im:
                    ph = imagehash.phash(im)
            except Exception:
                continue

            cand_bits = ph.hash.flatten().astype(np.int32)

            # Check distance against protected images
            dot = protected_bits @ cand_bits
            dists = protected_bits.sum(axis=1) + cand_bits.sum() - 2 * dot
            if dists.min() <= min_dist_threshold:
                continue

            # Check distance against already selected negatives
            if existing_neg_hashes:
                neg_mat = np.array(existing_neg_hashes)
                neg_dot = neg_mat @ cand_bits
                neg_dists = neg_mat.sum(axis=1) + cand_bits.sum() - 2 * neg_dot
                if neg_dists.min() <= 4:  # near-duplicate negative
                    continue

            existing_neg_hashes.append(cand_bits)
            accepted_for_domain.append(img_p)

        print(f"    -> Accepted {len(accepted_for_domain)} verified images.")
        category_summary[domain] = len(accepted_for_domain)
        for img_p in accepted_for_domain:
            selected_negatives.append((domain, img_p))

    # 3. Copy images to train and write empty label files
    print(f"\nInjecting {len(selected_negatives)} negative images into train split...")
    for domain, img_p in selected_negatives:
        clean_name = f"neg_{domain}_{img_p.stem}.jpg"
        out_img = train_img_dir / clean_name
        out_lbl = train_lbl_dir / f"neg_{domain}_{img_p.stem}.txt"

        shutil.copy2(img_p, out_img)
        # Empty label file (0 bytes) indicates background negative image in YOLO
        out_lbl.write_text("", encoding="utf-8")

    current_train_total = len(list(train_img_dir.glob("*.jpg")))
    neg_pct = (len(selected_negatives) / max(1, current_train_total)) * 100

    print("\nHard Negative Injection Complete:")
    print(f"  Total Hard Negatives Added: {len(selected_negatives)}")
    print(f"  Current Train Set Total: {current_train_total} images")
    print(f"  Negative Percentage in Train: {neg_pct:.1f}% (Target: 15-25%)")
    for d, c in category_summary.items():
        print(f"    - {d:20s}: {c} images")

    return {
        "added_count": len(selected_negatives),
        "category_summary": category_summary,
        "total_train_images": current_train_total,
        "negative_percentage": neg_pct
    }


def main():
    parser = argparse.ArgumentParser(description="Inject hard negatives into train split")
    parser.add_argument("--raw_neg_dir", type=str, default="ml/data/raw/hard_negatives")
    parser.add_argument("--merged_dir", type=str, default="ml/data/merged")
    parser.add_argument("--holdout_dir", type=str, default="ml/data/holdout")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Target roughly 450 images (15-25% of train set)
    target_counts = {
        "cctv_non_accident": 100,  # intersections, normal traffic flow
        "traffic_jam": 90,        # dense congestion, braking
        "parked_cars": 90,        # roadside parked vehicles
        "traffic_night": 85,      # night lighting, headlights, darkness
        "rain_traffic": 85,       # rain, wet road, reflections, glare
    }

    inject_hard_negatives(
        raw_neg_dir=Path(args.raw_neg_dir),
        merged_dir=Path(args.merged_dir),
        holdout_dir=Path(args.holdout_dir),
        target_counts=target_counts,
        min_dist_threshold=8,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
