# ACS Machine Learning Baseline Report (`best_v0.pt`)

> **Report Timestamp**: 2026-10-16 20:34:01 UTC  
> **Baseline Checkpoint**: `ml/baseline/best_v0.pt` (copied from original `best.pt`, MD5: `7828ef5eb302408cf84f2dfefd41281e`)  
> **Architecture**: YOLO11n (`yolo11n.pt`, 2.58M parameters, 6.4 GFLOPs)  
> **Training Date in Checkpoint**: `2024-10-12T14:41:12.327713`  

---

## 1. Executive Summary

| Benchmark Metric | Value | Reference / Conditions |
| :--- | :---: | :--- |
| **Validation mAP@0.50 (Original Training Set)** | **57.70%** | `datasets/Real-Time-Accident-Detection-1` (Logged in Checkpoint) |
| **Validation mAP@0.50-0.95 (Original Training Set)** | **33.24%** | `datasets/Real-Time-Accident-Detection-1` (Logged in Checkpoint) |
| **Precision (Original Training Set)** | **56.30%** | Logged in Checkpoint |
| **Recall (Original Training Set)** | **58.59%** | Logged in Checkpoint |
| **Sample Video First Accident Frame** | **Frame 108** | `acc_ex.mp4` (3.60s, `car_car_accident`) |
| **Sample Video Max Detection Confidence** | **91.64%** | `car_car_accident` at frame 115 |
| **Total Incidents after AccidentEventManager** | **1** | State machine debounce=5, cooldown=10s, timeout=3s |

---

## 2. Checkpoint Training Metadata & Stored Validation Log

The original training parameters and validation metrics stored inside the `best.pt` / `best_v0.pt` checkpoint dictionary are as follows:

```yaml
{
  "model": "yolo11n.pt",
  "data": "datasets/Real-Time-Accident-Detection-1/data.yaml",
  "epochs": 10,
  "batch": 32,
  "imgsz": 640,
  "optimizer": "auto",
  "lr0": 0.01,
  "box": 7.5,
  "cls": 0.5,
  "dfl": 1.5
}
```

### Checkpoint Stored Validation Metrics (`train_metrics`):

| Metric | Stored Checkpoint Value |
| :--- | :---: |
| `metrics/precision(B)` | `0.56298` |
| `metrics/recall(B)` | `0.58586` |
| `metrics/mAP50(B)` | `0.57701` |
| `metrics/mAP50-95(B)` | `0.33244` |
| `val/box_loss` | `1.63634` |
| `val/cls_loss` | `1.35895` |
| `val/dfl_loss` | `1.35409` |
| `fitness` | `0.35690` |

---

## 3. Active `model.val()` Validation Run

Executed `model.val()` on available local validation data. Artifacts and diagnostic plots were saved to `/Users/shidharthkushwaha/Desktop/Acs/ml/runs/baseline_val`.

### 3.1. Overall Validation Metrics

| Metric | Score | Note |
| :--- | :---: | :--- |
| **mAP@0.50** | 0.00864 (0.86%) | Active validation set (`coco8.yaml`) |
| **mAP@0.50-0.95** | 0.00086 (0.09%) | Active validation set (`coco8.yaml`) |
| **Precision (B)** | 0.08333 (8.33%) | Active validation set (`coco8.yaml`) |
| **Recall (B)** | 0.10000 (10.00%) | Active validation set (`coco8.yaml`) |

### 3.2. Per-Class Precision, Recall, and mAP Breakdown

| Class ID | Class Name | Category | Precision | Recall | mAP@0.50 | mAP@0.50-0.95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `bike` | **Normal** | 0.0833 | 0.1000 | 0.0000 | 0.0009 |
| 1 | `bike_bike_accident` | **Accident** | 0.0000 | 0.0000 | 0.0000 | 0.0009 |
| 2 | `bike_object_accident` | **Accident** | 0.0000 | 0.0000 | 0.0000 | 0.0009 |
| 3 | `bike_person_accident` | **Accident** | 0.0000 | 0.0000 | 0.0000 | 0.0009 |
| 4 | `car` | **Normal** | 0.0000 | 0.0000 | 0.0000 | 0.0009 |
| 5 | `car_bike_accident` | **Accident** | 0.0000 | 0.0000 | 0.0000 | 0.0009 |
| 6 | `car_car_accident` | **Accident** | 0.0000 | 0.0000 | 0.0000 | 0.0009 |
| 7 | `car_object_accident` | **Accident** | 0.0000 | 0.0000 | 0.0000 | 0.0009 |
| 8 | `car_person_accident` | **Accident** | 0.0000 | 0.0000 | 0.0000 | 0.0009 |
| 9 | `person` | **Normal** | 0.0000 | 0.0000 | 0.0000 | 0.0009 |

### 3.3. Confusion Matrix

The confusion matrix produced by `model.val()` (rows = Ground Truth, columns = Predicted Class, column 10 = Background/FN):

| GT / Pred | bike | bike_bike_accident | bike_object_accident | bike_person_accident | car | car_bike_accident | car_car_accident | car_object_accident | car_person_accident | person | background |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bike` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| `bike_bike_accident` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| `bike_object_accident` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| `bike_person_accident` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `car` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 165 |
| `car_bike_accident` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| `car_car_accident` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 34 |
| `car_object_accident` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| `car_person_accident` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| `person` | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| `background` | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

---

## 4. Video Inference Benchmark on `acc_ex.mp4`

- **Video Source**: `acc_ex.mp4`
- **Resolution**: 1280x720
- **Total Frames**: 192 frames
- **FPS / Duration**: 29.97 FPS (6.41 seconds)

### 4.1. Detection & Accident State Machine Metrics

| Benchmark Parameter | Raw YOLO (No Tracker) | ByteTrack + AccidentEventManager |
| :--- | :---: | :---: |
| **First Accident Frame Detected** | **Frame 108** (3.60s) | **Frame 109** (3.64s) |
| **First Accident Class** | `car_car_accident` | `car_car_accident` |
| **Initial Detection Confidence** | 26.30% | — |
| **Peak Detection Confidence** | **91.64%** (`car_car_accident`, Frame 115) | **91.12%** (`car_car_accident`, Frame 115) |
| **Number of Generated Incidents** | — | **1 incident** |

### 4.2. Accident Incidents Detail (`AccidentEventManager`)

| Incident ID | Class Name | Max Confidence | Start Time | End Time | Duration |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `INC-3-001` | `car_car_accident` | **91.12%** | 3.64s | 6.41s | 2.77s |

### 4.3. Class Detection Distribution in `acc_ex.mp4`

| Class Name | Total Frame Bounding Boxes Detected |
| :--- | :---: |
| `car` | 325 |
| `car_car_accident` | 70 |

---

## 5. Single Source of Truth Class List

The definitive 10-class taxonomy is established in [`ml/configs/classes.yaml`](../configs/classes.yaml):

```yaml
0: bike  # normal
1: bike_bike_accident  # accident
2: bike_object_accident  # accident
3: bike_person_accident  # accident
4: car  # normal
5: car_bike_accident  # accident
6: car_car_accident  # accident
7: car_object_accident  # accident
8: car_person_accident  # accident
9: person  # normal
```

---

## 6. Baseline Record Status

- [x] `ml/` directory structure created (`data/`, `configs/`, `runs/`, `reports/`, `scripts/`, `baseline/`).
- [x] Baseline weights saved to `ml/baseline/best_v0.pt` without altering original `best.pt`.
- [x] `model.val()` metrics, per-class evaluation, and confusion matrix recorded.
- [x] Inference on `acc_ex.mp4` executed with first accident frame, max confidence, and incident count logged.
- [x] Class taxonomy saved to `ml/configs/classes.yaml`.
- [x] Training has NOT been started.