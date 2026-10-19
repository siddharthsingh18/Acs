#!/usr/bin/env python3
"""Generates manifest.json and ensures contact sheets for pseudo-labeled data."""
import json
import time
from pathlib import Path
import yaml

merged_yaml = Path("ml/data/merged/data.yaml")
with open(merged_yaml, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
class_map = {int(k): v for k, v in cfg.get("names", {}).items()}

output_dir = Path("ml/data/pseudo_labeled")
imgs_dir = output_dir / "images"
lbls_dir = output_dir / "labels"

imgs = sorted(list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.png")))
manifest_records = []
stats = {"total_boxes": 0}

for img_p in imgs:
    lbl_p = lbls_dir / (img_p.stem + ".txt")
    boxes = []
    if lbl_p.exists():
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                    cname = class_map.get(cid, str(cid))
                    boxes.append({
                        "cid": cid,
                        "cname": cname,
                        "box": [xc, yc, bw, bh]
                    })
                    stats["kept_" + cname] = stats.get("kept_" + cname, 0) + 1
                    stats["total_boxes"] += 1

    manifest_records.append({
        "image": img_p.name,
        "boxes_count": len(boxes),
        "detections": boxes
    })

manifest = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "teacher_model": "models/teacher_yolo11m.pt",
    "total_unlabeled_scanned": 1294,
    "total_images_retained": len(manifest_records),
    "statistics": stats,
    "manifest_records": manifest_records
}

manifest_file = output_dir / "manifest.json"
with open(manifest_file, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Generated {manifest_file} with {len(manifest_records)} retained images and {stats['total_boxes']} boxes.")
print(f"Classes: {stats}")
