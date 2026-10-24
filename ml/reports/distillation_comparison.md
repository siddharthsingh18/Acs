# Knowledge Distillation Evaluation Report

> **UNVERIFIED:** Detection metrics and video rates conflict with other reports; the source dataset, clips, checkpoints, and raw evaluation artifacts are not available for reproduction.

## Student (YOLO11n + Pseudo-Labels) vs Current Production Model

**Generated**: 2026-10-19 19:01:40 UTC  
**Test Split**: Frozen `ml/data/merged/images/test` (417 images, 0 contamination)  

### 1. Test Split Detection Performance Matrix

| Model | Parameters | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Current Production (best.pt)** | 2,591,790 | 27.47% | 19.99% | 13.99% | 6.32% |
| **Distilled Student (YOLO11n)** | 2,591,790 | 35.73% | 28.40% | **32.39%** | 15.19% |
| **Net Change ($\Delta$)** | **0** | **+8.26%** | **+8.41%** | **+18.40%** | **+8.86%** |

### 2. Paired Bootstrap Confidence Intervals (B=200 resamples)

- **Overall Confidence (95% CI)**: Baseline `[0.7932, 0.8277]` vs Student `[0.6689, 0.7163]`
- **$\Delta$ Overall CI**: `[-0.1408, -0.0964]` (Mean: `-0.1187`)
- **Accident-Only $\Delta$ CI**: `[-0.1272, -0.0856]` (Mean: `-0.1059`)

### 3. Holdout Incident-Level Video Metrics

| Clip Name | Model | Incident Recall | False Alarms / Hour |
| :--- | :--- | :---: | :---: |
| `test_holdout_acc_ex.mp4` | Current Production | 100% | 561.94 FA/hr |
| `test_holdout_acc_ex.mp4` | Distilled Student | N/A | 0.00 FA/hr |
| `test_holdout_neg_traffic.mp4` | Current Production | N/A | 0.00 FA/hr |
| `test_holdout_neg_traffic.mp4` | Distilled Student | N/A | 0.00 FA/hr |
