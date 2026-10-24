#!/usr/bin/env python3
"""
ACS Knowledge Distillation - Teacher Model Fine-Tuning
------------------------------------------------------
Fine-tunes a larger teacher model (YOLO11m) on the merged dataset
(ml/data/merged/data.yaml) and reports its evaluation metrics on the frozen test split.
Checkpoints:
- ml/runs/teacher_yolo11m/weights/best.pt
- models/teacher_yolo11m.pt
Reports:
- ml/reports/teacher_metrics.json
- ml/reports/teacher_metrics.md
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acs.config import ACCIDENT_CLASSES, CLASS_NAMES
from acs.taxonomy import validate_model_taxonomy


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_teacher(
    data_yaml: Path = PROJECT_ROOT / "ml" / "data" / "merged" / "data.yaml",
    base_model: str = "yolo11m.pt",
    epochs: int = 1,
    batch: int = 16,
    imgsz: int = 512,
    freeze: int = 10,
    device: str = "auto",
    output_dir: Path = PROJECT_ROOT / "ml" / "runs" / "teacher_yolo11m",
    final_model_path: Path = PROJECT_ROOT / "models" / "teacher_yolo11m.pt",
    reports_dir: Path = PROJECT_ROOT / "ml" / "reports"
) -> Dict[str, Any]:
    """Fine-tunes teacher model on merged dataset and generates comprehensive metric reports."""
    start_time = time.time()
    resolved_dev = resolve_device() if device == "auto" else device
    print("=" * 80)
    print("🎓 TEACHER MODEL FINE-TUNING (YOLO11m)")
    print("=" * 80)
    print(f"Data: {data_yaml}")
    print(f"Base Weights: {base_model}")
    print(f"Epochs: {epochs} | Batch: {batch} | Imgsz: {imgsz} | Device: {resolved_dev}")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    model = YOLO(base_model)

    train_res = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=resolved_dev,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        freeze=freeze,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        verbose=True
    )

    best_weight_src = output_dir / "weights" / "best.pt"
    if not best_weight_src.exists():
        best_weight_src = output_dir / "weights" / "last.pt"

    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    if best_weight_src.exists():
        shutil.copy2(best_weight_src, final_model_path)
        print(f"✅ Saved teacher checkpoint to: {final_model_path}")

    # Evaluate Teacher on FROZEN TEST split
    print("\n🔍 Evaluating Teacher on FROZEN TEST Split...")
    teacher_eval_model = YOLO(str(final_model_path))
    validate_model_taxonomy(teacher_eval_model.names)
    val_res = teacher_eval_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        device=resolved_dev,
        plots=False,
        verbose=False
    )

    box = val_res.box
    class_map = teacher_eval_model.names
    ap_map = {}
    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        ap_map = {int(cid): idx for idx, cid in enumerate(box.ap_class_index)}

    per_class_metrics = []
    accident_p = []
    accident_r = []
    accident_map50 = []
    accident_map50_95 = []

    for cid in range(len(class_map)):
        cname = class_map[cid]
        is_acc = cname in ACCIDENT_CLASSES
        if cid in ap_map:
            arr_idx = ap_map[cid]
            p = float(box.p[arr_idx]) if hasattr(box, "p") and len(box.p) > arr_idx else 0.0
            r = float(box.r[arr_idx]) if hasattr(box, "r") and len(box.r) > arr_idx else 0.0
            map50 = float(box.ap50[arr_idx]) if hasattr(box, "ap50") and len(box.ap50) > arr_idx else 0.0
            map50_95 = float(box.ap[arr_idx]) if hasattr(box, "ap") and len(box.ap) > arr_idx else 0.0
        else:
            p = r = map50 = map50_95 = 0.0

        if is_acc:
            accident_p.append(p)
            accident_r.append(r)
            accident_map50.append(map50)
            accident_map50_95.append(map50_95)

        per_class_metrics.append({
            "class_id": cid,
            "class_name": cname,
            "is_accident": is_acc,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4)
        })

    teacher_metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_name": "YOLO11m (Teacher)",
        "model_parameters": sum(p.numel() for p in teacher_eval_model.model.parameters()),
        "checkpoint_path": str(final_model_path.relative_to(PROJECT_ROOT)),
        "dataset": str(data_yaml.relative_to(PROJECT_ROOT)),
        "split_evaluated": "test",
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4)
        },
        "accident_classes_macro": {
            "precision": round(float(np.mean(accident_p)), 4) if accident_p else 0.0,
            "recall": round(float(np.mean(accident_r)), 4) if accident_r else 0.0,
            "map50": round(float(np.mean(accident_map50)), 4) if accident_map50 else 0.0,
            "map50_95": round(float(np.mean(accident_map50_95)), 4) if accident_map50_95 else 0.0
        },
        "per_class": per_class_metrics,
        "training_time_seconds": round(time.time() - start_time, 2)
    }

    # Save reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "teacher_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(teacher_metrics, f, indent=2)

    md_path = reports_dir / "teacher_metrics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Teacher Model Fine-Tuning Metrics (YOLO11m)\n\n")
        f.write(f"- **Evaluated Model**: `{teacher_metrics['model_name']}` ({teacher_metrics['model_parameters']:,} parameters)\n")
        f.write(f"- **Dataset Evaluated**: `{teacher_metrics['dataset']}` (FROZEN TEST split, 417 images)\n")
        f.write(f"- **Training Duration**: {teacher_metrics['training_time_seconds']} seconds\n\n")
        f.write("### Overall Performance Matrix\n\n")
        f.write("| Metric Scope | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **All Classes** | {teacher_metrics['overall']['precision']*100:.2f}% | {teacher_metrics['overall']['recall']*100:.2f}% | {teacher_metrics['overall']['map50']*100:.2f}% | {teacher_metrics['overall']['map50_95']*100:.2f}% |\n")
        f.write(f"| **Accident Classes Only** | {teacher_metrics['accident_classes_macro']['precision']*100:.2f}% | {teacher_metrics['accident_classes_macro']['recall']*100:.2f}% | {teacher_metrics['accident_classes_macro']['map50']*100:.2f}% | {teacher_metrics['accident_classes_macro']['map50_95']*100:.2f}% |\n\n")
        f.write("### Per-Class Detailed Breakdown\n\n")
        f.write("| Class Name | Type | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for pc in per_class_metrics:
            badge = "🚨 Accident" if pc["is_accident"] else "🚗 Normal"
            f.write(f"| `{pc['class_name']}` | {badge} | {pc['precision']*100:.2f}% | {pc['recall']*100:.2f}% | {pc['map50']*100:.2f}% | {pc['map50_95']*100:.2f}% |\n")

    print(f"\n📊 Teacher metrics JSON: {json_path}")
    print(f"📄 Teacher report Markdown: {md_path}")
    print("=" * 80)
    return teacher_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    train_teacher(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device)
