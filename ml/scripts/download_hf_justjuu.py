#!/usr/bin/env python3
"""
ml/scripts/download_hf_justjuu.py
Downloads justjuu/traffic-accident-cctv-object-detection from Hugging Face
and converts it from Parquet format with COCO bboxes into standard YOLO txt format.
"""
import os
import io
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEST_DIR = os.path.join(ROOT_DIR, "ml", "data", "raw", "justjuu_cctv_accident_detection")

SPLITS_FILES = {
    "val": ["data/validation-00000-of-00001.parquet"],
    "test": ["data/test-00000-of-00001.parquet"],
    "train": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
}


def setup_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, "labels", split), exist_ok=True)


def convert_split(split_name, parquet_files):
    print(f"\n=== Processing split: {split_name} ===")
    total_converted = 0
    total_boxes = 0
    img_dir = os.path.join(DEST_DIR, "images", split_name)
    lbl_dir = os.path.join(DEST_DIR, "labels", split_name)

    global_idx = 0
    for pq_file in parquet_files:
        print(f"Downloading/fetching {pq_file}...")
        local_pq = hf_hub_download(
            repo_id="justjuu/traffic-accident-cctv-object-detection",
            filename=pq_file,
            repo_type="dataset",
        )
        print(f"Reading {local_pq}...")
        df = pd.read_parquet(local_pq)
        print(f"Loaded {len(df)} rows from {pq_file}")

        for _, row in df.iterrows():
            img_data = row["image"]
            objs = row["objects"]

            img_bytes = img_data.get("bytes")
            if not img_bytes:
                continue

            try:
                img = Image.open(io.BytesIO(img_bytes))
                w, h = img.size

                # Image filename
                img_name = f"cctv_{split_name}_{global_idx:05d}.jpg"
                lbl_name = f"cctv_{split_name}_{global_idx:05d}.txt"

                img_path = os.path.join(img_dir, img_name)
                lbl_path = os.path.join(lbl_dir, lbl_name)

                # Save image
                img.convert("RGB").save(img_path, "JPEG", quality=95)

                # Convert bboxes
                # In this dataset: bbox is [xmin, ymin, width, height]
                bboxes = objs.get("bbox", [])
                categories = objs.get("category", [])

                lines = []
                for bbox, cat_id in zip(bboxes, categories):
                    xmin, ymin, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    # Normalize
                    xc = (xmin + bw / 2.0) / w
                    yc = (ymin + bh / 2.0) / h
                    wn = bw / w
                    hn = bh / h

                    # Clamp to [0, 1]
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    wn = max(0.0, min(1.0, wn))
                    hn = max(0.0, min(1.0, hn))

                    lines.append(f"{int(cat_id)} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
                    total_boxes += 1

                with open(lbl_path, "w") as lf:
                    lf.write("\n".join(lines) + "\n")

                global_idx += 1
                total_converted += 1

            except Exception as e:
                print(f"Error processing row {global_idx}: {e}")

    print(f"Split {split_name} complete: {total_converted} images, {total_boxes} bounding boxes.")
    return total_converted, total_boxes


def create_data_yaml():
    yaml_content = f"""# CCTV Traffic Accident Object Detection Dataset (justjuu)
path: {DEST_DIR}
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: accident
  1: non_accident
"""
    with open(os.path.join(DEST_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)
    print("Created data.yaml")


def create_source_md(counts):
    md_content = f"""# Dataset Source: justjuu/traffic-accident-cctv-object-detection

- **URL**: https://huggingface.co/datasets/justjuu/traffic-accident-cctv-object-detection
- **Platform**: Hugging Face Hub
- **Version / Date**: 2024
- **License**: Creative Commons Zero v1.0 Universal (CC0-1.0 - Public Domain Dedication)
- **Image Count**: {sum(counts.values())} total images
  - Train: {counts.get('train', 0)} images
  - Validation: {counts.get('val', 0)} images
  - Test: {counts.get('test', 0)} images
- **Class Names (2 classes)**:
  - `0`: `accident`
  - `1`: `non_accident`
- **Format**: Converted from Hugging Face Parquet (COCO bounding box `[xmin, ymin, width, height]`) into standard YOLOv11 normalized `[cls_id, x_center, y_center, width, height]` text files with companion JPEG images.
- **Problems / Mapping Considerations**:
  - The dataset distinguishes between `accident` and `non_accident` (normal traffic/vehicles).
  - All accident types (car-car, car-bike, motorcycle) are labeled under a single unified `accident` category.
  - Excellent for CCTV accident discrimination and low-resolution traffic camera generalization.
"""
    with open(os.path.join(DEST_DIR, "SOURCE.md"), "w") as f:
        f.write(md_content)
    print("Created SOURCE.md")


def main():
    setup_dirs()
    counts = {}
    for split in ["val", "test", "train"]:
        n_imgs, _ = convert_split(split, SPLITS_FILES[split])
        counts[split] = n_imgs

    create_data_yaml()
    create_source_md(counts)
    print("\n✅ Successfully processed justjuu/traffic-accident-cctv-object-detection!")


if __name__ == "__main__":
    main()
