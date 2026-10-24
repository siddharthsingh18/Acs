#!/usr/bin/env python3
"""
Automated CCTV Accident Detection System (ACS)
Rare Class Oversampling Script for YOLOv11 Training (Experiment D)

Analyzes class imbalance in the training split, identifies rare classes
(accident classes and underrepresented vehicle/person classes), and creates
an augmented training set by oversampling images containing rare instances.
Maintains identical validation and test splits with zero leakage.
"""

import shutil
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import yaml
from acs.config import ACCIDENT_CLASSES, CLASS_NAMES, NORMAL_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_train_distribution(train_labels_dir: Path, target_classes: List[str]) -> Tuple[Counter, Dict[int, List[Path]]]:
    """Scans all label files in the training set and maps classes to image labels."""
    class_counts = Counter()
    class_to_files = {i: [] for i in range(len(target_classes))}

    label_files = sorted(list(train_labels_dir.glob("*.txt")))
    for lbl_p in label_files:
        content = lbl_p.read_text(encoding="utf-8").strip()
        if not content:
            continue  # Hard negative (empty label)

        file_classes = set()
        for line in content.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(parts[0])
            if 0 <= cid < len(target_classes):
                class_counts[cid] += 1
                file_classes.add(cid)

        for cid in file_classes:
            class_to_files[cid].append(lbl_p)

    return class_counts, class_to_files


def create_oversampled_dataset(
    source_dir: Path,
    output_dir: Path,
    classes_yaml_path: Path,
    target_min_instances: int = 1500,
    accident_multiplier: int = 4
) -> Path:
    """
    Creates an oversampled train split in output_dir, referencing original val/test splits.
    Returns the path to the newly generated data.yaml.
    """
    classes_cfg = load_yaml(classes_yaml_path)
    target_classes = CLASS_NAMES

    src_train_imgs = source_dir / "images" / "train"
    src_train_lbls = source_dir / "labels" / "train"
    src_val_imgs = source_dir / "images" / "val"
    src_test_imgs = source_dir / "images" / "test"

    out_train_imgs = output_dir / "images" / "train"
    out_train_lbls = output_dir / "labels" / "train"

    # Reset output training directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    out_train_imgs.mkdir(parents=True, exist_ok=True)
    out_train_lbls.mkdir(parents=True, exist_ok=True)

    # Symlink val and test splits for zero overhead and identical evaluation
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Copy or symlink val and test
    for split in ["val", "test"]:
        split_img_src = source_dir / "images" / split
        split_lbl_src = source_dir / "labels" / split
        split_img_dst = output_dir / "images" / split
        split_lbl_dst = output_dir / "labels" / split

        split_img_dst.symlink_to(split_img_src.resolve(), target_is_directory=True)
        split_lbl_dst.symlink_to(split_lbl_src.resolve(), target_is_directory=True)

    print("Analyzing baseline training class distribution...")
    initial_counts, class_to_files = analyze_train_distribution(src_train_lbls, target_classes)

    print("\nInitial Training Box Counts:")
    for cid, cname in enumerate(target_classes):
        cnt = initial_counts.get(cid, 0)
        print(f"  Class {cid:2d} ({cname:22s}): {cnt:6d} boxes ({len(class_to_files[cid])} images)")

    # 1. First, symlink or copy all base images & labels to output
    print("\nPopulating base training images into oversampled directory...")
    all_train_imgs = list(src_train_imgs.glob("*.jpg"))
    for img_p in all_train_imgs:
        lbl_p = src_train_lbls / f"{img_p.stem}.txt"
        shutil.copy2(img_p, out_train_imgs / img_p.name)
        if lbl_p.exists():
            shutil.copy2(lbl_p, out_train_lbls / lbl_p.name)
        else:
            (out_train_lbls / f"{img_p.stem}.txt").touch()

    # 2. Oversample rare classes
    # Define accident classes
    accident_cids = {
        class_id for class_id, class_name in enumerate(target_classes)
        if class_name in ACCIDENT_CLASSES
    }
    minority_cids = {
        class_id for class_id, class_name in enumerate(target_classes)
        if class_name in NORMAL_CLASSES and class_name in {"bike", "person"}
    }

    oversampled_instances = Counter(initial_counts)
    oversample_copies = 0

    print("\nCalculating oversampling multipliers for rare classes...")
    for cid in range(len(target_classes)):
        cname = target_classes[cid]
        current_cnt = initial_counts.get(cid, 0)
        if current_cnt == 0:
            continue

        if cid in accident_cids:
            # High priority: accident classes get aggressive oversampling
            mult = accident_multiplier
        elif cid in minority_cids and current_cnt < target_min_instances:
            # Minority non-accident classes (bike, person)
            mult = max(1, min(6, round(target_min_instances / max(current_cnt, 1))))
        else:
            mult = 1

        if mult > 1:
            matching_lbls = class_to_files[cid]
            print(f"  Oversampling class '{cname}' (Class {cid}): {current_cnt} -> target {current_cnt * mult} ({mult}x replication across {len(matching_lbls)} images)")

            for copy_idx in range(1, mult):
                for lbl_p in matching_lbls:
                    img_name = f"{lbl_p.stem}.jpg"
                    img_p = src_train_imgs / img_name
                    if not img_p.exists():
                        continue

                    new_stem = f"{lbl_p.stem}_os_c{cid}_v{copy_idx}"
                    new_img_p = out_train_imgs / f"{new_stem}.jpg"
                    new_lbl_p = out_train_lbls / f"{new_stem}.txt"

                    if not new_img_p.exists():
                        shutil.copy2(img_p, new_img_p)
                        shutil.copy2(lbl_p, new_lbl_p)
                        oversample_copies += 1

    # Re-audit final oversampled distribution
    final_counts, _ = analyze_train_distribution(out_train_lbls, target_classes)
    total_imgs_after = len(list(out_train_imgs.glob("*.jpg")))

    print(f"\nOversampling Complete: Added {oversample_copies} replicated samples. Total Train Images: {total_imgs_after}")
    print("\nFinal Oversampled Class Distribution:")
    for cid, cname in enumerate(target_classes):
        before = initial_counts.get(cid, 0)
        after = final_counts.get(cid, 0)
        gain = f"+{after - before}" if after > before else "0"
        print(f"  Class {cid:2d} ({cname:22s}): {before:6d} -> {after:6d} ({gain:>6s})")

    # Write data.yaml for oversampled dataset
    out_yaml_path = output_dir / "data.yaml"
    data_dict = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(target_classes),
        "names": {i: name for i, name in enumerate(target_classes)}
    }

    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)

    print(f"\nGenerated oversampled data configuration: {out_yaml_path}")
    return out_yaml_path


if __name__ == "__main__":
    src = PROJECT_ROOT / "ml" / "data" / "merged"
    dst = PROJECT_ROOT / "ml" / "data" / "oversampled"
    cfg = PROJECT_ROOT / "ml" / "configs" / "classes.yaml"
    create_oversampled_dataset(src, dst, cfg)
