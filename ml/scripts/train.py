#!/usr/bin/env python3
"""
Automated CCTV Accident Detection System (ACS)
Master Model Training Script (Experiments A, B, C, D)

Features:
- Two-Phase Fine-Tuning: Phase 1 (freeze=10 for 10 epochs), Phase 2 (unfreeze for 90 epochs).
- Loss Divergence Monitor: Automatically halts and reports if val/cls_loss diverges (> 5.0 or > 3.5x train loss).
- Deterministic Seeding: Enforces seed=42 and deterministic=True.
- Hardware Detection: Resolves 'cuda' -> 'mps' -> 'cpu' and calculates CPU runtime warning if > 6 hours.
- Canonical Checkpointing: Guarantees last.pt and best.pt per run in ml/runs/<experiment_name>/weights/.
- Experiments Comparison: Generates ml/runs/experiments_summary.md with comprehensive metrics table.
"""

import os
import sys
import time
import shutil
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import yaml
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42, deterministic: bool = True):
    """Enforces reproducible seeding across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(requested: str = "auto") -> Tuple[str, bool, str]:
    """Resolves target hardware device and returns (device_name, is_cpu, warning_msg)."""
    if requested != "auto":
        dev = requested.lower()
        is_cpu = dev == "cpu"
        return dev, is_cpu, f"Manually requested device: {dev}"

    if torch.cuda.is_available():
        return "cuda", False, "Detected CUDA GPU acceleration"
    elif torch.backends.mps.is_available():
        return "mps", False, "Detected Apple Silicon MPS acceleration (Metal)"
    else:
        return "cpu", True, "No GPU acceleration detected. Defaulting to CPU."


def check_cpu_runtime_warning(n_train_images: int, imgsz: int, epochs: int, threshold_hours: float = 6.0) -> Optional[str]:
    """Calculates estimated CPU training runtime and returns warning if it exceeds threshold."""
    # Approximate CPU speed for YOLO detection at 960px: ~0.12 - 0.20s per image forward/backward
    est_sec_per_img = 0.15
    est_total_sec = n_train_images * est_sec_per_img * epochs
    est_hours = est_total_sec / 3600.0

    if est_hours > threshold_hours:
        return (
            f"⚠️ [HARDWARE WARNING] Training on CPU is estimated to take {est_hours:.1f} hours "
            f"(exceeds user-specified threshold of {threshold_hours:.1f} hours). "
            f"Recommended: Run on CUDA GPU or Apple Silicon MPS device."
        )
    return None


class LossDivergenceMonitor:
    """
    Ultralytics callback hook that monitors val/cls_loss after each validation epoch.
    If loss spikes or diverges from training loss, it stops training and generates an alert report.
    """
    def __init__(
        self,
        max_cls_loss: float = 5.0,
        max_ratio: float = 3.5,
        min_epochs_before_ratio: int = 3,
        run_dir: Optional[Path] = None
    ):
        self.max_cls_loss = max_cls_loss
        self.max_ratio = max_ratio
        self.min_epochs_before_ratio = min_epochs_before_ratio
        self.run_dir = run_dir
        self.diverged = False
        self.divergence_details = {}

    def __call__(self, trainer):
        # Reads recorded metrics from trainer.csv at on_fit_epoch_end
        if not hasattr(trainer, "csv") or not trainer.csv.exists():
            return

        try:
            lines = trainer.csv.read_text(encoding="utf-8").strip().splitlines()
            if len(lines) < 2:
                return

            header = [h.strip() for h in lines[0].split(",")]
            last_row = [v.strip() for v in lines[-1].split(",")]
            metric_map = {k: float(v) for k, v in zip(header, last_row) if v.replace(".", "", 1).replace("-", "", 1).isdigit()}

            epoch = int(metric_map.get("epoch", trainer.epoch + 1))
            val_cls = metric_map.get("val/cls_loss", None)
            train_cls = metric_map.get("train/cls_loss", None)

            if val_cls is None:
                return

            # Check 1: Absolute ceiling threshold
            if val_cls > self.max_cls_loss:
                self.diverged = True
                self.divergence_details = {
                    "epoch": epoch,
                    "val_cls_loss": val_cls,
                    "train_cls_loss": train_cls,
                    "trigger": f"val/cls_loss ({val_cls:.4f}) exceeded absolute threshold ({self.max_cls_loss:.4f})",
                    "suggestion": "Instability detected: lower lr0 or increase freeze layers."
                }
                trainer.stop = True
                self._write_report()
                print(f"\n🛑 [LOSS DIVERGENCE DETECTED] Epoch {epoch}: {self.divergence_details['trigger']}")
                return

            # Check 2: Ratio of val/cls_loss to train/cls_loss after warmup
            if epoch >= self.min_epochs_before_ratio and train_cls is not None and train_cls > 0:
                ratio = val_cls / train_cls
                if ratio > self.max_ratio:
                    self.diverged = True
                    self.divergence_details = {
                        "epoch": epoch,
                        "val_cls_loss": val_cls,
                        "train_cls_loss": train_cls,
                        "trigger": f"val/cls_loss ({val_cls:.4f}) is {ratio:.2f}x train/cls_loss ({train_cls:.4f}), exceeding ratio limit ({self.max_ratio:.2f}x)",
                        "suggestion": "Instability detected: lower lr0 or increase freeze layers."
                    }
                    trainer.stop = True
                    self._write_report()
                    print(f"\n🛑 [LOSS DIVERGENCE DETECTED] Epoch {epoch}: {self.divergence_details['trigger']}")
                    return

        except Exception as e:
            # Silent fallback to not crash training loop on minor parsing anomalies
            pass

    def _write_report(self):
        if self.run_dir:
            report_file = self.run_dir / "DIVERGENCE_ALERT.md"
            content = [
                "# 🛑 Loss Divergence Alert Report",
                f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Epoch**: {self.divergence_details.get('epoch', 'N/A')}",
                f"**Trigger**: {self.divergence_details.get('trigger', '')}",
                f"**Validation Cls Loss**: `{self.divergence_details.get('val_cls_loss', 0.0):.4f}`",
                f"**Training Cls Loss**: `{self.divergence_details.get('train_cls_loss', 0.0):.4f}`",
                "",
                "> [!CAUTION]",
                f"> {self.divergence_details.get('suggestion', '')}",
                "",
                "Training was automatically halted to preserve model weights and prevent degenerate gradient updates."
            ]
            report_file.write_text("\n".join(content), encoding="utf-8")


def run_training_experiment(
    exp_key: str,
    cfg: dict,
    device: str,
    override_epochs: Optional[int] = None,
    custom_data_yaml: Optional[Path] = None,
    extra_augment: Optional[dict] = None
) -> Dict[str, Any]:
    """Executes a full two-phase training experiment."""
    global_cfg = cfg["global"]
    div_cfg = cfg.get("divergence", {})
    exp_cfg = cfg[exp_key]

    exp_name = exp_cfg["name"]
    project_dir = PROJECT_ROOT / global_cfg.get("project", "ml/runs")
    project_dir.mkdir(parents=True, exist_ok=True)
    exp_run_dir = project_dir / exp_name
    exp_run_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = str(custom_data_yaml or (PROJECT_ROOT / global_cfg["data"]).resolve())
    imgsz = exp_cfg.get("imgsz", 960)
    batch = exp_cfg.get("batch", -1)
    cos_lr = exp_cfg.get("cos_lr", True)
    lr0 = exp_cfg.get("lr0", 0.005)
    optimizer = exp_cfg.get("optimizer", "AdamW")
    patience = exp_cfg.get("patience", 20)
    seed = global_cfg.get("seed", 42)
    deterministic = global_cfg.get("deterministic", True)

    total_epochs = override_epochs or exp_cfg.get("total_epochs", 100)
    phase1_epochs = min(exp_cfg.get("phase1", {}).get("epochs", 10), total_epochs)
    phase2_epochs = max(0, total_epochs - phase1_epochs)

    print("\n" + "=" * 80)
    print(f"🚀 STARTING EXPERIMENT: {exp_name.upper()}")
    print("=" * 80)
    print(f"Description: {exp_cfg.get('description', '')}")
    print(f"Base Model: {exp_cfg['model']}")
    print(f"Target Device: {device.upper()}")
    print(f"Image Size: {imgsz}px | Batch: {batch} | Optimizer: {optimizer} | lr0: {lr0}")
    print(f"Schedule: Total {total_epochs} ep (Phase 1: {phase1_epochs} ep freeze=10 -> Phase 2: {phase2_epochs} ep unfrozen)")
    print(f"Data Source: {data_yaml}")
    print(f"Run Output: {exp_run_dir}")

    # Save resolved hyperparameters
    resolved_hp = {
        "experiment": exp_key,
        "name": exp_name,
        "base_model": exp_cfg["model"],
        "device": device,
        "imgsz": imgsz,
        "batch": batch,
        "optimizer": optimizer,
        "lr0": lr0,
        "cos_lr": cos_lr,
        "patience": patience,
        "seed": seed,
        "deterministic": deterministic,
        "total_epochs": total_epochs,
        "phase1_epochs": phase1_epochs,
        "phase1_freeze": exp_cfg.get("phase1", {}).get("freeze", 10),
        "phase2_epochs": phase2_epochs,
        "phase2_freeze": exp_cfg.get("phase2", {}).get("freeze", 0),
        "extra_augment": extra_augment or {},
        "data_yaml": data_yaml,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(exp_run_dir / "hyperparameters.yaml", "w", encoding="utf-8") as f:
        yaml.dump(resolved_hp, f, default_flow_style=False)

    # --------------------------------------------------------------------------
    # PHASE 1: Frozen Backbone Head Adaptation
    # --------------------------------------------------------------------------
    print(f"\n[PHASE 1] Initializing {phase1_epochs} epochs with freeze={exp_cfg['phase1']['freeze']} (lr0={lr0})...")
    p1_name = f"{exp_name}_phase1"
    model = YOLO(exp_cfg["model"])

    monitor_p1 = LossDivergenceMonitor(
        max_cls_loss=div_cfg.get("max_cls_loss", 5.0),
        max_ratio=div_cfg.get("cls_loss_ratio", 3.5),
        min_epochs_before_ratio=div_cfg.get("min_epochs_before_ratio", 3),
        run_dir=exp_run_dir
    )
    model.add_callback("on_fit_epoch_end", monitor_p1)

    t0 = time.time()
    train_args_p1 = {
        "data": data_yaml,
        "epochs": phase1_epochs,
        "freeze": exp_cfg["phase1"]["freeze"],
        "lr0": lr0,
        "cos_lr": cos_lr,
        "imgsz": imgsz,
        "batch": batch,
        "optimizer": optimizer,
        "project": str(project_dir),
        "name": p1_name,
        "seed": seed,
        "deterministic": deterministic,
        "device": device,
        "save": True,
        "val": True,
        "plots": True,
        "verbose": True
    }
    if extra_augment:
        train_args_p1.update(extra_augment)

    try:
        model.train(**train_args_p1)
    except Exception as e:
        print(f"Error in Phase 1 training: {e}")

    p1_dir = project_dir / p1_name
    p1_best = p1_dir / "weights" / "best.pt"
    p1_last = p1_dir / "weights" / "last.pt"

    if monitor_p1.diverged:
        print(f"\n🛑 Experiment halted due to loss divergence in Phase 1.")
        return {
            "experiment": exp_key,
            "status": "DIVERGED",
            "divergence_details": monitor_p1.divergence_details,
            "best_pt": None,
            "last_pt": None,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }

    # --------------------------------------------------------------------------
    # PHASE 2: Full Network Fine-Tuning (Unfrozen)
    # --------------------------------------------------------------------------
    checkpoint_to_load = p1_best if p1_best.exists() else p1_last
    if not checkpoint_to_load or not checkpoint_to_load.exists():
        checkpoint_to_load = Path(exp_cfg["model"])

    p2_name = f"{exp_name}_phase2"
    final_best_pt = None
    final_last_pt = None

    if phase2_epochs > 0:
        print(f"\n[PHASE 2] Loading checkpoint {checkpoint_to_load} for {phase2_epochs} epochs unfrozen (freeze=0, patience={patience})...")
        model2 = YOLO(str(checkpoint_to_load))

        monitor_p2 = LossDivergenceMonitor(
            max_cls_loss=div_cfg.get("max_cls_loss", 5.0),
            max_ratio=div_cfg.get("cls_loss_ratio", 3.5),
            min_epochs_before_ratio=div_cfg.get("min_epochs_before_ratio", 3),
            run_dir=exp_run_dir
        )
        model2.add_callback("on_fit_epoch_end", monitor_p2)

        train_args_p2 = {
            "data": data_yaml,
            "epochs": phase2_epochs,
            "freeze": 0,
            "lr0": lr0,
            "cos_lr": cos_lr,
            "patience": patience,
            "imgsz": imgsz,
            "batch": batch,
            "optimizer": optimizer,
            "project": str(project_dir),
            "name": p2_name,
            "seed": seed,
            "deterministic": deterministic,
            "device": device,
            "save": True,
            "val": True,
            "plots": True,
            "verbose": True
        }
        if extra_augment:
            train_args_p2.update(extra_augment)

        try:
            model2.train(**train_args_p2)
        except Exception as e:
            print(f"Error in Phase 2 training: {e}")

        p2_dir = project_dir / p2_name
        final_best_pt = p2_dir / "weights" / "best.pt"
        final_last_pt = p2_dir / "weights" / "last.pt"

        if monitor_p2.diverged:
            print(f"\n🛑 Experiment halted due to loss divergence in Phase 2.")
            return {
                "experiment": exp_key,
                "status": "DIVERGED",
                "divergence_details": monitor_p2.divergence_details,
                "best_pt": str(p1_best),
                "last_pt": str(final_last_pt),
                "mAP50": 0.0,
                "mAP50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
    else:
        final_best_pt = p1_best
        final_last_pt = p1_last

    total_training_time = time.time() - t0

    # Ensure canonical weights directory in exp_run_dir
    weights_dir = exp_run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    if final_best_pt and final_best_pt.exists():
        shutil.copy2(final_best_pt, weights_dir / "best.pt")
    if final_last_pt and final_last_pt.exists():
        shutil.copy2(final_last_pt, weights_dir / "last.pt")

    # --------------------------------------------------------------------------
    # FINAL VALIDATION EVALUATION
    # --------------------------------------------------------------------------
    best_weight_eval = weights_dir / "best.pt"
    if not best_weight_eval.exists():
        best_weight_eval = Path(exp_cfg["model"])

    print(f"\n[VALIDATION EVALUATION] Evaluating best checkpoint: {best_weight_eval} at imgsz={imgsz}...")
    eval_model = YOLO(str(best_weight_eval))
    val_metrics = eval_model.val(data=data_yaml, split="val", imgsz=imgsz, device=device, verbose=False)

    p_box = float(val_metrics.results_dict.get("metrics/precision(B)", 0.0))
    r_box = float(val_metrics.results_dict.get("metrics/recall(B)", 0.0))
    map50 = float(val_metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(val_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))

    result = {
        "experiment": exp_key,
        "name": exp_name,
        "status": "COMPLETED",
        "device": device,
        "total_time_sec": round(total_training_time, 1),
        "best_weights": str(weights_dir / "best.pt"),
        "last_weights": str(weights_dir / "last.pt"),
        "precision": round(p_box, 4),
        "recall": round(r_box, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "fitness": round(float(val_metrics.fitness or 0.0), 4)
    }

    print("\n" + "-" * 60)
    print(f"✨ EXPERIMENT {exp_key.upper()} COMPLETED SUCCESSFULLY!")
    print(f"Precision: {p_box*100:.2f}% | Recall: {r_box*100:.2f}% | mAP50: {map50*100:.2f}% | mAP50-95: {map50_95*100:.2f}%")
    print(f"Canonical Weights: {weights_dir / 'best.pt'}")
    print("-" * 60)

    return result


def write_master_summary(results: List[Dict[str, Any]], summary_path: Path):
    """Writes ml/runs/experiments_summary.md comparing all executed experiments."""
    lines = []
    lines.append("# Automated CCTV Accident Detection System (ACS)")
    lines.append("## YOLOv11 Training Experiments Summary & Comparative Analysis\n")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append("**Logging Directory**: `ml/runs/`  \n")

    lines.append("### 1. Comparative Performance Matrix\n")
    lines.append("| Experiment | Description | Status | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) | Best Weights |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    for r in results:
        status_icon = "✅" if r.get("status") == "COMPLETED" else "🛑"
        lines.append(
            f"| **{r.get('name', r.get('experiment'))}** | {r.get('experiment')} | {status_icon} {r.get('status')} | "
            f"{r.get('precision', 0.0)*100:.2f}% | {r.get('recall', 0.0)*100:.2f}% | "
            f"**{r.get('mAP50', 0.0)*100:.2f}%** | {r.get('mAP50_95', 0.0)*100:.2f}% | "
            f"`{r.get('best_weights', 'N/A')}` |"
        )

    lines.append("\n---\n")
    lines.append("### 2. Experiment Highlights & Design Decisions\n")
    lines.append("- **Experiment A (YOLO11n Two-Phase)**: Evaluates fine-tuning stability from `best_v0.pt` with 10 initial frozen epochs.")
    lines.append("- **Experiment B (YOLO11s Two-Phase)**: Assesses recall scaling using higher capacity YOLO11s architecture.")
    lines.append("- **Experiment C (Stronger Augmentation)**: Tests mosaic (1.0), mixup (0.1), color jitter, scale, and flipping to prevent overfitting.")
    lines.append("- **Experiment D (Oversampling)**: Addresses rare class imbalance via targeted replication of minority classes without touching holdout sets.")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nGenerated master experiments summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Master YOLOv11 Training Pipeline (ACS)")
    parser.add_argument("--config", type=str, default="ml/configs/train.yaml", help="Path to training config YAML")
    parser.add_argument("--experiment", type=str, default="A", choices=["A", "B", "C", "D", "all"], help="Experiment to run")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('cuda', 'mps', 'cpu', 'auto')")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs for quick validation/dry-run")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    cfg = load_yaml(cfg_path)

    global_cfg = cfg["global"]
    set_seed(global_cfg.get("seed", 42), global_cfg.get("deterministic", True))

    target_device, is_cpu, dev_msg = resolve_device(args.device if args.device != "auto" else global_cfg.get("device", "auto"))
    print(f"\n[DEVICE INITIALIZATION] {dev_msg} (Selected: '{target_device}')")

    if is_cpu:
        n_imgs = 2338  # Merged training set size
        target_epochs = args.epochs or cfg.get("exp_a", {}).get("total_epochs", 100)
        warn = check_cpu_runtime_warning(
            n_train_images=n_imgs,
            imgsz=cfg.get("exp_a", {}).get("imgsz", 960),
            epochs=target_epochs,
            threshold_hours=cfg.get("hardware", {}).get("cpu_max_estimated_hours", 6.0)
        )
        if warn:
            print(f"\n{warn}\n")

    executed_results = []

    # Run Sequence
    experiments_to_run = ["exp_a", "exp_b", "exp_c", "exp_d"] if args.experiment == "all" else [f"exp_{args.experiment.lower()}"]

    for exp_key in experiments_to_run:
        if exp_key == "exp_a":
            res = run_training_experiment("exp_a", cfg, target_device, override_epochs=args.epochs)
            executed_results.append(res)

        elif exp_key == "exp_b":
            res = run_training_experiment("exp_b", cfg, target_device, override_epochs=args.epochs)
            executed_results.append(res)

        elif exp_key == "exp_c":
            # Select best of A or B based on mAP50
            completed_prev = [r for r in executed_results if r.get("status") == "COMPLETED" and r["experiment"] in ["exp_a", "exp_b"]]
            if completed_prev:
                best_prev = max(completed_prev, key=lambda x: x.get("mAP50", 0.0))
                best_model_weights = best_prev["best_weights"]
                print(f"\n[EXP C SELECTION] Dynamically selected {best_prev['name']} as base (mAP50: {best_prev['mAP50']*100:.2f}%)")
            else:
                best_model_weights = cfg["exp_a"]["model"]

            cfg["exp_c"]["model"] = best_model_weights
            extra_aug = cfg["exp_c"].get("augment", {})
            res = run_training_experiment("exp_c", cfg, target_device, override_epochs=args.epochs, extra_augment=extra_aug)
            executed_results.append(res)

        elif exp_key == "exp_d":
            # Oversampling
            print("\n[EXP D PREPARATION] Running oversampling script for rare classes...")
            from ml.scripts.oversample import create_oversampled_dataset
            os_data_yaml = create_oversampled_dataset(
                source_dir=PROJECT_ROOT / "ml" / "data" / "merged",
                output_dir=PROJECT_ROOT / "ml" / "data" / "oversampled",
                classes_yaml_path=PROJECT_ROOT / "ml" / "configs" / "classes.yaml"
            )

            # Pick best of A, B, or C
            completed_prev = [r for r in executed_results if r.get("status") == "COMPLETED"]
            if completed_prev:
                best_overall = max(completed_prev, key=lambda x: x.get("mAP50", 0.0))
                base_model = best_overall["best_weights"]
            else:
                base_model = cfg["exp_a"]["model"]

            cfg["exp_d"]["model"] = base_model
            res = run_training_experiment("exp_d", cfg, target_device, override_epochs=args.epochs, custom_data_yaml=os_data_yaml)
            executed_results.append(res)

    # Master Summary
    summary_path = PROJECT_ROOT / "ml" / "runs" / "experiments_summary.md"
    write_master_summary(executed_results, summary_path)


if __name__ == "__main__":
    main()
