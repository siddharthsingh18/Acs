#!/usr/bin/env python3
"""
Build Distillation Training Dataset (Student Dataset)
----------------------------------------------------
Combines real training dataset with teacher pseudo-labels under strict constraints:
1. The pseudo-labeled share is CAPPED at 50% of the combined training data:
   N_pseudo / (N_real + N_pseudo) <= 0.50  ==>  N_pseudo <= N_real.
2. NEVER touches or mixes pseudo-labels into val or test splits.
   val and test splits strictly point to ml/data/merged/images/val and test.
3. Generates dataset descriptor: ml/data/distillation/data.yaml.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acs.taxonomy import load_classes


def build_distillation_dataset(
    real_dataset_yaml: Path = PROJECT_ROOT / "ml" / "data" / "merged" / "data.yaml",
    pseudo_labels_dir: Path = PROJECT_ROOT / "ml" / "data" / "pseudo_labeled",
    output_dir: Path = PROJECT_ROOT / "ml" / "data" / "distillation",
    max_pseudo_share: float = 0.50,
    seed: int = 42
) -> Dict[str, Any]:
    """Combines real training data and teacher pseudo-labels with 50% cap."""
    random.seed(seed)
    print("=" * 80)
    print("📦 BUILDING DISTILLATION TRAINING DATASET")
    print("=" * 80)

    train_img_out = output_dir / "images" / "train"
    train_lbl_out = output_dir / "labels" / "train"
    train_img_out.mkdir(parents=True, exist_ok=True)
    train_lbl_out.mkdir(parents=True, exist_ok=True)

    # 1. Load and copy real training set
    real_train_imgs = list((PROJECT_ROOT / "ml" / "data" / "merged" / "images" / "train").glob("*.jpg"))
    real_train_lbls_dir = PROJECT_ROOT / "ml" / "data" / "merged" / "labels" / "train"

    print(f"Copying {len(real_train_imgs)} real training samples...")
    for img_p in real_train_imgs:
        dst_img = train_img_out / img_p.name
        if not dst_img.exists():
            shutil.copy2(img_p, dst_img)

        lbl_p = real_train_lbls_dir / (img_p.stem + ".txt")
        if lbl_p.exists():
            dst_lbl = train_lbl_out / lbl_p.name
            if not dst_lbl.exists():
                shutil.copy2(lbl_p, dst_lbl)

    n_real = len(real_train_imgs)

    # 2. Add pseudo-labels capped at max_pseudo_share (50% => N_pseudo <= N_real)
    max_allowed_pseudo = int((max_pseudo_share / (1.0 - max_pseudo_share)) * n_real)
    pseudo_imgs = list((pseudo_labels_dir / "images").glob("*.jpg")) + list((pseudo_labels_dir / "images").glob("*.png"))
    pseudo_lbls_dir = pseudo_labels_dir / "labels"

    # Sort or shuffle deterministically
    random.shuffle(pseudo_imgs)
    pseudo_selected = pseudo_imgs[:max_allowed_pseudo]
    n_pseudo_used = len(pseudo_selected)

    actual_share = (n_pseudo_used / (n_real + n_pseudo_used)) if (n_real + n_pseudo_used) > 0 else 0.0
    print(f"Candidate pseudo-labels available: {len(pseudo_imgs)}")
    print(f"Capping constraint: max {max_allowed_pseudo} pseudo samples (current actual share: {actual_share * 100:.2f}%)")

    for img_p in pseudo_selected:
        dst_img = train_img_out / f"pseudo_{img_p.name}"
        if not dst_img.exists():
            shutil.copy2(img_p, dst_img)

        lbl_p = pseudo_lbls_dir / (img_p.stem + ".txt")
        if lbl_p.exists():
            dst_lbl = train_lbl_out / f"pseudo_{img_p.stem}.txt"
            if not dst_lbl.exists():
                shutil.copy2(lbl_p, dst_lbl)

    total_train = n_real + n_pseudo_used
    print(f"\n✅ Total Combined Training Images: {total_train} (Real: {n_real}, Pseudo: {n_pseudo_used}, Share: {actual_share * 100:.2f}%)")

    # 3. Create dataset descriptor pointing to frozen val and test splits
    data_yaml_path = output_dir / "data.yaml"
    classes = load_classes()
    dataset_cfg = {
        "path": str(output_dir.resolve()),
        "train": str(train_img_out.resolve()),
        "val": str((PROJECT_ROOT / "ml" / "data" / "merged" / "images" / "val").resolve()),
        "test": str((PROJECT_ROOT / "ml" / "data" / "merged" / "images" / "test").resolve()),
        "nc": len(classes),
        "names": {item["id"]: item["name"] for item in classes},
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_cfg, f, sort_keys=False)

    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "real_samples_count": n_real,
        "pseudo_samples_count": n_pseudo_used,
        "total_train_samples": total_train,
        "pseudo_share_percentage": round(actual_share * 100, 2),
        "data_yaml": str(data_yaml_path.relative_to(PROJECT_ROOT)),
        "frozen_val_dir": "ml/data/merged/images/val",
        "frozen_test_dir": "ml/data/merged/images/test"
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"📄 Distillation dataset YAML written to: {data_yaml_path}")
    print("=" * 80)
    return meta


if __name__ == "__main__":
    build_distillation_dataset()
