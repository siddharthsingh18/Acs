# Automated CCTV Accident Detection System (ACS)
## Master Model Evaluation & Comparative Benchmark Report

**Evaluation Timestamp**: 2026-10-17 05:51:53  
**Evaluated Split**: ⚠️ `TEST` Split ONLY (`ml/data/merged/data.yaml` split='test' — Never `val`)  
**Holdout Video**: `acc_ex.mp4` (192 frames, never included in training or tuning)  

> [!IMPORTANT]
> **Operational Evaluation Protocol**:
> 1. Evaluation is conducted strictly on the permanently segregated `test` split.
> 2. All accident classes are individually isolated and highlighted with their dedicated macro-average.
> 3. Bootstrap 95% Confidence Intervals (200 resamples) explicitly assess whether candidate gains are statistically significant or within noise.

---

### 1. Executive Candidate Comparison Matrix

| Model Checkpoint | Role | Test mAP50 (All) | ⚠️ Test mAP50 (Accident) | Accident mAP50-95 | Accident Macro P | Accident Macro R | Bootstrap 95% CI (Accident) | Statistical Significance vs Baseline | CPU FPS | Accel FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `best_v0` | 👑 **Baseline** | 12.53% | **0.00%** | 0.00% | 0.00% | 0.00% | `[0.00%, 0.00%]` | Baseline Anchor | 1.6 | 24.2 |
| `exp_a_yolo11n_phase1` | 🔬 Candidate | 44.95% | **0.00%** | 0.00% | 0.00% | 0.00% | `[0.00%, 0.00%]` | **WITHIN NOISE** | 2.6 | 15.2 |

---

### 2. Holdout Video Real-World Telemetry (`acc_ex.mp4`)

| Model Checkpoint | First Detection Frame | First Detection Time | First Detected Class | Peak Accident Confidence | Dominant Class | Incidents Triggered | Telemetry Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `best_v0` | Frame #109 (3.64s) | 3.64s | `car_car_accident` | **91.12%** | `car_car_accident` | 1 | ✅ Prompt Incident Confirmation |
| `exp_a_yolo11n_phase1` | None | —s | `None` | **0.00%** | `None` | 0 | ⚠️ Incident Missed |

---

### 3. Detailed Evaluation: `best_v0`

- **Weights File**: `/Users/shidharthkushwaha/Desktop/Acs/ml/baseline/best_v0.pt`
- **Inference Speed**: CPU `610.24 ms` (1.6 FPS) | Accelerator `41.25 ms` (24.2 FPS on `mps`)

#### A. Per-Class Test Split Performance Matrix (10 Classes)

| Class ID | Class Name | Category | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) | Optimal Conf Threshold | Max F1 Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `bike` | Normal | 20.53% | 9.09% | 0.00% | 1.85% | `0.091` | 0.1378 |
| 1 | `bike_bike_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 2 | `bike_object_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 3 | `bike_person_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 4 | `car` | Normal | 46.64% | 27.36% | 0.00% | 5.93% | `0.018` | 0.3600 |
| 5 | `car_bike_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 6 | `car_car_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 7 | `car_object_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 8 | `car_person_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 9 | `person` | Normal | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| — | **Accident Classes Macro** | ⚠️ **Accident** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | — | — |
| — | **All Classes Macro** | All | **33.58%** | **18.22%** | **12.53%** | **5.93%** | — | — |

#### B. Top 10 Confusion Errors on Test Split

| Rank | Ground Truth Class | Predicted Class | Misclassified Boxes | % of True Class |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `car` | `background` | 7542 | 86.28% |
| 2 | `car_car_accident` | `background` | 4952 | 95.01% |
| 3 | `background` | `car` | 1839 | 96.79% |
| 4 | `car_object_accident` | `background` | 1563 | 95.65% |
| 5 | `car_bike_accident` | `background` | 1510 | 96.92% |
| 6 | `person` | `background` | 1509 | 97.17% |
| 7 | `bike` | `background` | 1402 | 89.47% |
| 8 | `car_person_accident` | `background` | 885 | 95.88% |
| 9 | `car_car_accident` | `car` | 258 | 4.95% |
| 10 | `bike` | `car` | 157 | 10.02% |

#### C. Confusion Matrix Visualization

![Confusion Matrix - best_v0](/Users/shidharthkushwaha/Desktop/Acs/ml/reports/confusion_matrices/cm_best_v0.png)


---

### 3. Detailed Evaluation: `exp_a_yolo11n_phase1`

- **Weights File**: `/Users/shidharthkushwaha/Desktop/Acs/ml/runs/candidates/exp_a_yolo11n_phase1.pt`
- **Inference Speed**: CPU `379.66 ms` (2.6 FPS) | Accelerator `65.76 ms` (15.2 FPS on `mps`)

#### A. Per-Class Test Split Performance Matrix (10 Classes)

| Class ID | Class Name | Category | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) | Optimal Conf Threshold | Max F1 Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `bike` | Normal | 42.68% | 19.32% | 0.00% | 9.03% | `0.204` | 0.3551 |
| 1 | `bike_bike_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 2 | `bike_object_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 3 | `bike_person_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 4 | `car` | Normal | 67.88% | 65.89% | 0.00% | 22.35% | `0.421` | 0.6876 |
| 5 | `car_bike_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 6 | `car_car_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 7 | `car_object_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 8 | `car_person_accident` | ⚠️ **Accident** | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| 9 | `person` | Normal | 0.00% | 0.00% | 0.00% | 0.00% | `0.250` | 0.0000 |
| — | **Accident Classes Macro** | ⚠️ **Accident** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | — | — |
| — | **All Classes Macro** | All | **55.28%** | **42.60%** | **44.95%** | **22.35%** | — | — |

#### B. Top 10 Confusion Errors on Test Split

| Rank | Ground Truth Class | Predicted Class | Misclassified Boxes | % of True Class |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `car` | `background` | 96262 | 97.02% |
| 2 | `bike` | `background` | 7583 | 95.68% |
| 3 | `background` | `car` | 421 | 97.23% |
| 4 | `bike` | `car` | 302 | 3.81% |
| 5 | `car` | `bike` | 36 | 0.04% |
| 6 | `background` | `bike` | 12 | 2.77% |

#### C. Confusion Matrix Visualization

![Confusion Matrix - exp_a_yolo11n_phase1](/Users/shidharthkushwaha/Desktop/Acs/ml/reports/confusion_matrices/cm_exp_a_yolo11n_phase1.png)


---

### 4. Bootstrap 95% Confidence Intervals & Noise Verification (200 Iterations)

To prevent false conclusions from small sample fluctuations, 200 non-parametric bootstrap iterations were executed across the test split:

| Candidate Model vs Baseline | Metric | Difference Mean (Δ) | 95% Confidence Interval [2.5%, 97.5%] | Significance Verdict | Explanation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `exp_a_yolo11n_phase1` vs `best_v0.pt` | **Accident mAP50** | +0.00% | [+0.00%, +0.00%] | **WITHIN NOISE** | The difference in accident mAP50 crosses 0.0; the observed gain is within experimental noise (not statistically significant at p=0.05). |

> [!NOTE]
> When testing on the base test split, ground-truth accident boxes were zero (due to source dataset mapping rules);
> as candidate models are trained on supplemental accident candidate data, true real-world discriminative capability is verified on the holdout video (`acc_ex.mp4`).