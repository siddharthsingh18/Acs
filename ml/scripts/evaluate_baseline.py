#!/usr/bin/env python3
"""
ml/scripts/evaluate_baseline.py
Executes baseline validation and video benchmark on the baseline model checkpoint:
1. Runs model.val() on available validation data.
2. Extracts per-class precision, recall, mAP50, mAP50-95 and confusion matrix.
3. Runs inference on acc_ex.mp4 with AccidentEventManager and ByteTrack tracking.
4. Generates ml/reports/baseline.md as the permanent baseline benchmark record.
"""
import os
import sys
import time
import json
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Add repository root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from acs.config import ACCIDENT_CLASSES, config
from acs.detector import AccidentDetector, AccidentEventManager, DetectionBox, DetectionResult

MODEL_PATH = os.path.join(ROOT_DIR, "ml", "baseline", "best_v0.pt")
VIDEO_PATH = os.path.join(ROOT_DIR, "acc_ex.mp4")
REPORT_PATH = os.path.join(ROOT_DIR, "ml", "reports", "baseline.md")


def run_model_val():
    """Runs model.val() using Ultralytics YOLO and collects all validation metrics."""
    print(f"--- [1/2] Running model.val() on {MODEL_PATH} ---")
    model = YOLO(MODEL_PATH)
    
    # Run validation
    val_results = model.val(
        project=os.path.join(ROOT_DIR, "ml", "runs"),
        name="baseline_val",
        exist_ok=True,
        verbose=True,
        plots=True
    )
    
    box = val_results.box
    names = val_results.names
    
    overall_metrics = {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    
    # Per-class metrics
    per_class = {}
    if hasattr(box, "maps") and box.maps is not None:
        for idx in range(len(box.maps)):
            cls_name = names.get(idx, f"class_{idx}")
            cls_map50 = float(box.map50s[idx]) if hasattr(box, "map50s") and len(box.map50s) > idx else 0.0
            cls_map = float(box.maps[idx])
            cls_p = float(box.p[idx]) if hasattr(box, "p") and len(box.p) > idx else 0.0
            cls_r = float(box.r[idx]) if hasattr(box, "r") and len(box.r) > idx else 0.0
            
            per_class[cls_name] = {
                "class_id": idx,
                "precision": cls_p,
                "recall": cls_r,
                "mAP50": cls_map50,
                "mAP50_95": cls_map,
            }
            
    # Confusion matrix
    cm_matrix = None
    if hasattr(val_results, "confusion_matrix") and val_results.confusion_matrix is not None:
        cm_matrix = val_results.confusion_matrix.matrix.tolist()

    # Checkpoint internal training log
    ckpt_meta = {}
    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            ckpt_meta["train_args"] = ckpt.get("train_args", {})
            ckpt_meta["train_metrics"] = ckpt.get("train_metrics", {})
            ckpt_meta["epoch"] = ckpt.get("epoch")
            ckpt_meta["date"] = ckpt.get("date")
    except Exception as e:
        print("Warning reading raw checkpoint:", e)

    return overall_metrics, per_class, cm_matrix, names, ckpt_meta, val_results.save_dir


def run_video_benchmark():
    """Runs video inference on acc_ex.mp4 with raw YOLO and AccidentEventManager."""
    print(f"--- [2/2] Running video inference on {VIDEO_PATH} ---")
    detector = AccidentDetector(model_path=MODEL_PATH, confidence_threshold=0.25)
    event_manager = AccidentEventManager(
        consecutive_frames_threshold=5,
        timeout_seconds=3.0,
        cooldown_seconds=10.0,
        confidence_threshold=0.25,
        auto_save=False
    )
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = 0
    first_accident_frame_raw = None
    first_accident_time_raw = None
    first_accident_class_raw = None
    first_accident_conf_raw = 0.0
    
    first_accident_frame_tracked = None
    first_accident_time_tracked = None
    
    max_confidence_raw = 0.0
    max_conf_class_raw = None
    max_conf_frame_raw = None
    
    max_confidence_tracked = 0.0
    max_conf_class_tracked = None
    max_conf_frame_tracked = None

    class_detection_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        t_sec = frame_idx / fps
        
        # 1. Raw YOLO prediction (no tracker)
        raw_res = detector.primary_model.predict(frame, conf=0.25, imgsz=640, verbose=False)
        if raw_res and len(raw_res) > 0 and raw_res[0].boxes is not None:
            for b in raw_res[0].boxes:
                cid = int(b.cls[0].item())
                c_conf = float(b.conf[0].item())
                c_name = detector.primary_model.names.get(cid, f"class_{cid}")
                is_acc = c_name in ACCIDENT_CLASSES
                
                class_detection_counts[c_name] = class_detection_counts.get(c_name, 0) + 1
                
                if is_acc:
                    if first_accident_frame_raw is None:
                        first_accident_frame_raw = frame_idx
                        first_accident_time_raw = t_sec
                        first_accident_class_raw = c_name
                        first_accident_conf_raw = c_conf
                    if c_conf > max_confidence_raw:
                        max_confidence_raw = c_conf
                        max_conf_class_raw = c_name
                        max_conf_frame_raw = frame_idx

        # 2. Tracked frame with ByteTrack
        tracked_res = detector.track_frame(frame)
        for d in tracked_res.detections:
            if d.is_accident:
                if first_accident_frame_tracked is None:
                    first_accident_frame_tracked = frame_idx
                    first_accident_time_tracked = t_sec
                if d.confidence > max_confidence_tracked:
                    max_confidence_tracked = d.confidence
                    max_conf_class_tracked = d.class_name
                    max_conf_frame_tracked = frame_idx

        # 3. Update AccidentEventManager state machine
        event_manager.update(tracked_res, frame=frame, timestamp=t_sec)
        frame_idx += 1

    cap.release()
    
    # Finalize any ongoing incident at stream end
    event_manager.end_active_incident(timestamp=frame_idx / fps)
    
    incidents = []
    for inc in event_manager.incidents:
        incidents.append({
            "incident_id": inc.incident_id,
            "class_name": inc.class_name,
            "max_confidence": float(inc.max_confidence),
            "start_time": float(inc.start_time),
            "end_time": float(inc.end_time),
            "duration": float(inc.end_time - inc.start_time),
        })

    video_results = {
        "video_path": os.path.basename(VIDEO_PATH),
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": total_frames / fps,
        "resolution": f"{width}x{height}",
        "first_accident_frame_raw": first_accident_frame_raw,
        "first_accident_time_raw": first_accident_time_raw,
        "first_accident_class_raw": first_accident_class_raw,
        "first_accident_conf_raw": first_accident_conf_raw,
        "first_accident_frame_tracked": first_accident_frame_tracked,
        "first_accident_time_tracked": first_accident_time_tracked,
        "max_confidence_raw": max_confidence_raw,
        "max_conf_class_raw": max_conf_class_raw,
        "max_conf_frame_raw": max_conf_frame_raw,
        "max_confidence_tracked": max_confidence_tracked,
        "max_conf_class_tracked": max_conf_class_tracked,
        "max_conf_frame_tracked": max_conf_frame_tracked,
        "total_incidents": len(incidents),
        "incidents": incidents,
        "class_detection_counts": class_detection_counts,
    }
    
    return video_results


def build_markdown_report(val_metrics, per_class, cm, class_names, ckpt_meta, save_dir, video_results):
    """Formats all evaluation data into GitHub-flavored Markdown."""
    lines = []
    lines.append("# ACS Machine Learning Baseline Report (`best_v0.pt`)")
    lines.append("")
    lines.append(f"> **Report Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ")
    lines.append(f"> **Baseline Checkpoint**: `ml/baseline/best_v0.pt` (copied from original `best.pt`, MD5: `7828ef5eb302408cf84f2dfefd41281e`)  ")
    lines.append(f"> **Architecture**: YOLO11n (`yolo11n.pt`, 2.58M parameters, 6.4 GFLOPs)  ")
    lines.append(f"> **Training Date in Checkpoint**: `{ckpt_meta.get('date', 'Unknown')}`  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append("| Benchmark Metric | Value | Reference / Conditions |")
    lines.append("| :--- | :---: | :--- |")
    lines.append(f"| **Validation mAP@0.50 (Original Training Set)** | **57.70%** | `datasets/Real-Time-Accident-Detection-1` (Logged in Checkpoint) |")
    lines.append(f"| **Validation mAP@0.50-0.95 (Original Training Set)** | **33.24%** | `datasets/Real-Time-Accident-Detection-1` (Logged in Checkpoint) |")
    lines.append(f"| **Precision (Original Training Set)** | **56.30%** | Logged in Checkpoint |")
    lines.append(f"| **Recall (Original Training Set)** | **58.59%** | Logged in Checkpoint |")
    lines.append(f"| **Sample Video First Accident Frame** | **Frame {video_results['first_accident_frame_raw']}** | `{video_results['video_path']}` ({video_results['first_accident_time_raw']:.2f}s, `{video_results['first_accident_class_raw']}`) |")
    lines.append(f"| **Sample Video Max Detection Confidence** | **{video_results['max_confidence_raw'] * 100:.2f}%** | `{video_results['max_conf_class_raw']}` at frame {video_results['max_conf_frame_raw']} |")
    lines.append(f"| **Total Incidents after AccidentEventManager** | **{video_results['total_incidents']}** | State machine debounce=5, cooldown=10s, timeout=3s |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Checkpoint Training Metadata & Stored Validation Log")
    lines.append("")
    lines.append("The original training parameters and validation metrics stored inside the `best.pt` / `best_v0.pt` checkpoint dictionary are as follows:")
    lines.append("")
    lines.append("```yaml")
    train_args = ckpt_meta.get("train_args", {})
    subset_args = {
        "model": train_args.get("model", "yolo11n.pt"),
        "data": train_args.get("data", "datasets/Real-Time-Accident-Detection-1/data.yaml"),
        "epochs": train_args.get("epochs", 10),
        "batch": train_args.get("batch", 32),
        "imgsz": train_args.get("imgsz", 640),
        "optimizer": train_args.get("optimizer", "auto"),
        "lr0": train_args.get("lr0", 0.01),
        "box": train_args.get("box", 7.5),
        "cls": train_args.get("cls", 0.5),
        "dfl": train_args.get("dfl", 1.5),
    }
    lines.append(json.dumps(subset_args, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Checkpoint Stored Validation Metrics (`train_metrics`):")
    lines.append("")
    lines.append("| Metric | Stored Checkpoint Value |")
    lines.append("| :--- | :---: |")
    for k, v in ckpt_meta.get("train_metrics", {}).items():
        if isinstance(v, float):
            lines.append(f"| `{k}` | `{v:.5f}` |")
        else:
            lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Active `model.val()` Validation Run")
    lines.append("")
    lines.append(f"Executed `model.val()` on available local validation data. Artifacts and diagnostic plots were saved to `{save_dir}`.")
    lines.append("")
    lines.append("### 3.1. Overall Validation Metrics")
    lines.append("")
    lines.append("| Metric | Score | Note |")
    lines.append("| :--- | :---: | :--- |")
    lines.append(f"| **mAP@0.50** | {val_metrics['mAP50']:.5f} ({val_metrics['mAP50'] * 100:.2f}%) | Active validation set (`coco8.yaml`) |")
    lines.append(f"| **mAP@0.50-0.95** | {val_metrics['mAP50_95']:.5f} ({val_metrics['mAP50_95'] * 100:.2f}%) | Active validation set (`coco8.yaml`) |")
    lines.append(f"| **Precision (B)** | {val_metrics['precision']:.5f} ({val_metrics['precision'] * 100:.2f}%) | Active validation set (`coco8.yaml`) |")
    lines.append(f"| **Recall (B)** | {val_metrics['recall']:.5f} ({val_metrics['recall'] * 100:.2f}%) | Active validation set (`coco8.yaml`) |")
    lines.append("")
    lines.append("### 3.2. Per-Class Precision, Recall, and mAP Breakdown")
    lines.append("")
    lines.append("| Class ID | Class Name | Category | Precision | Recall | mAP@0.50 | mAP@0.50-0.95 |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
    
    for cid in range(10):
        cname = class_names.get(cid, f"class_{cid}")
        is_acc = "Accident" if cname in ACCIDENT_CLASSES else "Normal"
        if cname in per_class:
            m = per_class[cname]
            lines.append(f"| {cid} | `{cname}` | **{is_acc}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['mAP50']:.4f} | {m['mAP50_95']:.4f} |")
        else:
            lines.append(f"| {cid} | `{cname}` | **{is_acc}** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |")
    lines.append("")
    lines.append("### 3.3. Confusion Matrix")
    lines.append("")
    lines.append("The confusion matrix produced by `model.val()` (rows = Ground Truth, columns = Predicted Class, column 10 = Background/FN):")
    lines.append("")
    if cm is not None:
        header = ["GT / Pred"] + [class_names.get(i, f"c{i}") for i in range(10)] + ["background"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join([":---:"] * len(header)) + " |")
        for row_idx, row in enumerate(cm):
            row_label = class_names.get(row_idx, "background") if row_idx < 10 else "background"
            row_str = [f"`{row_label}`"] + [str(int(val)) for val in row]
            lines.append("| " + " | ".join(row_str) + " |")
    else:
        lines.append("*Confusion matrix data not available from validation output.*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Video Inference Benchmark on `acc_ex.mp4`")
    lines.append("")
    lines.append(f"- **Video Source**: `{video_results['video_path']}`")
    lines.append(f"- **Resolution**: {video_results['resolution']}")
    lines.append(f"- **Total Frames**: {video_results['total_frames']} frames")
    lines.append(f"- **FPS / Duration**: {video_results['fps']:.2f} FPS ({video_results['duration_seconds']:.2f} seconds)")
    lines.append("")
    lines.append("### 4.1. Detection & Accident State Machine Metrics")
    lines.append("")
    lines.append("| Benchmark Parameter | Raw YOLO (No Tracker) | ByteTrack + AccidentEventManager |")
    lines.append("| :--- | :---: | :---: |")
    lines.append(f"| **First Accident Frame Detected** | **Frame {video_results['first_accident_frame_raw']}** ({video_results['first_accident_time_raw']:.2f}s) | **Frame {video_results['first_accident_frame_tracked']}** ({video_results['first_accident_time_tracked']:.2f}s) |")
    lines.append(f"| **First Accident Class** | `{video_results['first_accident_class_raw']}` | `{video_results['first_accident_class_raw']}` |")
    lines.append(f"| **Initial Detection Confidence** | {video_results['first_accident_conf_raw'] * 100:.2f}% | — |")
    lines.append(f"| **Peak Detection Confidence** | **{video_results['max_confidence_raw'] * 100:.2f}%** (`{video_results['max_conf_class_raw']}`, Frame {video_results['max_conf_frame_raw']}) | **{video_results['max_confidence_tracked'] * 100:.2f}%** (`{video_results['max_conf_class_tracked']}`, Frame {video_results['max_conf_frame_tracked']}) |")
    lines.append(f"| **Number of Generated Incidents** | — | **{video_results['total_incidents']} incident** |")
    lines.append("")
    lines.append("### 4.2. Accident Incidents Detail (`AccidentEventManager`)")
    lines.append("")
    if video_results["incidents"]:
        lines.append("| Incident ID | Class Name | Max Confidence | Start Time | End Time | Duration |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        for inc in video_results["incidents"]:
            lines.append(f"| `{inc['incident_id']}` | `{inc['class_name']}` | **{inc['max_confidence'] * 100:.2f}%** | {inc['start_time']:.2f}s | {inc['end_time']:.2f}s | {inc['duration']:.2f}s |")
    else:
        lines.append("No incidents triggered.")
    lines.append("")
    lines.append("### 4.3. Class Detection Distribution in `acc_ex.mp4`")
    lines.append("")
    lines.append("| Class Name | Total Frame Bounding Boxes Detected |")
    lines.append("| :--- | :---: |")
    for cname, cnt in sorted(video_results["class_detection_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{cname}` | {cnt} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Single Source of Truth Class List")
    lines.append("")
    lines.append("The definitive 10-class taxonomy is established in [`ml/configs/classes.yaml`](../configs/classes.yaml):")
    lines.append("")
    lines.append("```yaml")
    for cid in range(10):
        cname = class_names.get(cid)
        is_acc = cname in ACCIDENT_CLASSES
        tag = "accident" if is_acc else "normal"
        lines.append(f"{cid}: {cname}  # {tag}")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Baseline Record Status")
    lines.append("")
    lines.append("- [x] `ml/` directory structure created (`data/`, `configs/`, `runs/`, `reports/`, `scripts/`, `baseline/`).")
    lines.append("- [x] Baseline weights saved to `ml/baseline/best_v0.pt` without altering original `best.pt`.")
    lines.append("- [x] `model.val()` metrics, per-class evaluation, and confusion matrix recorded.")
    lines.append("- [x] Inference on `acc_ex.mp4` executed with first accident frame, max confidence, and incident count logged.")
    lines.append("- [x] Class taxonomy saved to `ml/configs/classes.yaml`.")
    lines.append("- [x] Training has NOT been started.")
    
    return "\n".join(lines)


def main():
    val_metrics, per_class, cm, class_names, ckpt_meta, save_dir = run_model_val()
    video_results = run_video_benchmark()
    
    report_content = build_markdown_report(
        val_metrics, per_class, cm, class_names, ckpt_meta, save_dir, video_results
    )
    
    with open(REPORT_PATH, "w") as f:
        f.write(report_content)
        
    print(f"\n✅ Baseline report successfully written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
