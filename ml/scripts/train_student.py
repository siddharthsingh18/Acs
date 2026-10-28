#!/usr/bin/env python3
"""
ACS Knowledge Distillation - Student Model Training (YOLO11n)
-----------------------------------------------------------
Trains the compact, deployable student model (YOLO11n) on combined
real training data plus teacher pseudo-labels (capped at <= 50%).
Checkpoints:
- ml/runs/student_yolo11n/weights/best.pt
- models/student_yolo11n.pt
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_student(
    data_yaml: Path = PROJECT_ROOT / "ml" / "data" / "distillation" / "data.yaml",
    base_model: str = "best.pt",
    epochs: int = 2,
    batch: int = 16,
    imgsz: int = 512,
    device: str = "auto",
    freeze: int = 10,
    output_dir: Path = PROJECT_ROOT / "ml" / "runs" / "student_yolo11n",
    final_model_path: Path = PROJECT_ROOT / "models" / "student_yolo11n.pt"
) -> Path:
    """Trains the nano student model on the distillation dataset."""
    start_time = time.time()
    resolved_dev = resolve_device() if device == "auto" else device
    if not Path(base_model).exists() and base_model == "best.pt":
        base_model = "yolo11n.pt"

    print("=" * 80)
    print("🚀 STUDENT MODEL TRAINING (YOLO11n)")
    print("=" * 80)
    print(f"Data: {data_yaml}")
    print(f"Base Model: {base_model}")
    print(f"Epochs: {epochs} | Batch: {batch} | Imgsz: {imgsz} | Freeze: {freeze} | Device: {resolved_dev}")

    torch.manual_seed(42)
    np.random.seed(42)

    model = YOLO(base_model)

    train_res = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        freeze=freeze,
        device=resolved_dev,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        seed=42,
        deterministic=True,
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
        print(f"✅ Saved student checkpoint to: {final_model_path}")

    elapsed = round(time.time() - start_time, 2)
    print(f"Student training finished in {elapsed}s.")
    return final_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="best.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--freeze", type=int, default=10)
    args = parser.parse_args()
    train_student(
        base_model=args.base_model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        freeze=args.freeze
    )
