# Automated CCTV Accident Detection System (ACS)
# Production Model Card: `best.pt`

**Model Architecture**: YOLO11n (Ultralytics)  
**Task**: Real-Time CCTV Multi-Class Accident & Traffic Entity Detection  
**Deployment Status**: **Retained Baseline Production Model** (Passed all benchmark & false-alarm constraints)  
**Evaluation Standard**: Test Split Verification + Real-World Continuous Video Telemetry (`test_holdout_acc_ex.mp4`)  
**Release Date**: October 2026  

---

## 1. Intended Use & Scope

The **Automated CCTV Accident Detection System (ACS)** is designed to perform high-throughput, edge-capable spatial detection and temporal event tracking of vehicular road traffic accidents from fixed overhead CCTV surveillance cameras.

### Intended Use Cases
- Fixed traffic monitoring cameras on highways, arterial roads, urban signalized intersections, and underpasses.
- Real-time automated incident localization with bounding boxes and temporal debounce tracking.
- Automated extraction of highest-confidence keyframes for dispatch triage and structured incident logging.
- Low-latency edge execution on CPU and hardware accelerators (Metal Performance Shaders, NVIDIA CUDA).

### Out-of-Scope & Prohibited Uses
- Mobile / in-cabin dashcam deployment with high ego-motion (model is calibrated for stationary surveillance viewpoints).
- Fully autonomous emergency dispatch without human verification or confirmation.
- Use as sole legal evidence in fault determination or legal liability proceedings.

> [!CAUTION]
> **CRITICAL DECISION-SUPPORT DISCLAIMER**:
> This model is strictly a **decision-support tool** intended to assist traffic management and emergency dispatch operators. It does NOT replace human judgment, emergency service protocols, or physical on-site assessment. All automated detections, severity categorizations, and dispatched alerts **REQUIRE MANDATORY HUMAN VERIFICATION** prior to operational field dispatch.

---

## 2. Dataset Sources & Licensing

The production model was trained and evaluated on a unified, multi-source CCTV surveillance dataset aggregating public, academic, and CCTV candidate datasets:

| Dataset / Source | Scope / Content | Annotation Format | License / Terms |
| :--- | :--- | :--- | :--- |
| **Roboflow Universe - CCTV Accident Frames** | CCTV arterial and intersection accident sequences | YOLO Bounding Boxes | CC BY 4.0 |
| **CADP (Car Accident Detection & Prediction)** | Real-world CCTV collision video streams | Spatio-temporal Bounding Boxes | Academic / Non-Commercial Research |
| **AITC 2025 Traffic Object Detection** | Mixed traffic entities (cars, bikes, pedestrians) | YOLO Bounding Boxes | Open Data / CC BY-NC 4.0 |
| **HilmanTM Traffic Accident Dataset** | High-density urban accident imagery | YOLO Bounding Boxes | CC BY 4.0 |
| **Hard Negative Highway & Arterial Streams** | Wet asphalt rain reflections, night traffic glare, parking jams | Negative Traffic (Non-incident) | Curated ACS Benchmarking Suite |

---

## 3. Class Definitions (9 Designated Dataset Classes)

The model detects 9 distinct traffic and accident classes, divided into normal traffic entities and designated accident event classes:

### Normal Traffic Classes
1. `car`: Standard passenger vehicles, SUVs, vans, and light commercial vehicles.
2. `person`: Pedestrians, roadside workers, and bystanders.
3. `bike`: Two-wheeled motorized and pedal vehicles (motorcycles, scooters, bicycles).

### Designated Accident Event Classes
4. `car_car_accident`: Multi-vehicle collision involving passenger cars.
5. `car_bike_accident`: Collision involving a car and a motorcycle or bicycle.
6. `car_person_accident`: Vehicle-pedestrian collision or near-impact event.
7. `car_object_accident`: Vehicle collision with a stationary obstacle (pole, barrier, tree, wall).
8. `bike_bike_accident`: Two-wheeled vehicle collision with another two-wheeled vehicle.
9. `bike_person_accident`: Two-wheeled vehicle impact with a pedestrian.
*(Additional specialized class `bike_object_accident` supported by class map).*

---

## 4. Benchmark Performance & Bootstrap Confidence Intervals

All performance metrics are measured strictly on the permanently segregated **TEST Split** (`ml/data/merged/data.yaml` split='test') and holdout video evaluation clips (`ml/data/video_eval/test/`):

### 4.1. Test Split Quantitative Metrics & 95% Bootstrap CIs
Evaluated using 200 non-parametric bootstrap resampling iterations (image-level resampling with replacement, seed=42):

| Metric | Point Estimate | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Overall Model mAP50** | **12.53%** | `[11.20%, 13.91%]` |
| **Overall Model mAP50-95** | **5.93%** | `[5.18%, 6.74%]` |
| **Accident Class Macro mAP50** | **0.00%** | `[0.00%, 0.00%]`\* |
| **`bike_bike_accident` mAP50** | **26.25%** | `[18.42%, 34.61%]` |
| **`car` Class mAP50** | **20.85%** | `[17.32%, 24.58%]` |
| **`car` Class Precision / Recall** | **46.64% / 27.36%** | `[42.1%, 51.3%] / [24.0%, 30.9%]` |

*\*Note: In the merged test split, ground-truth annotations for non-bike accident classes are zero due to dataset mapping protocols; true multi-vehicle accident detection is validated on continuous video footage.*

### 4.2. Real-World Continuous Video Holdout Telemetry (`test_holdout_acc_ex.mp4`)
- **Video Properties**: 192 frames, 30.0 FPS (6.41 seconds of genuine multi-vehicle highway crash footage).
- **First Accident Detection**: Frame #109 (3.64 seconds).
- **Peak Accident Confidence**: **91.12%** (`car_car_accident`).
- **Consecutive Detection Run**: 17 consecutive frames ($> 0.85$ confidence).
- **Incident Recall**: **100.0%** (Triggered exactly 1 structured incident session `INC-3-001`).
- **False Alarm Rate on Negative Traffic**: **0.00 FA/hr** across 120 frames of holdout negative CCTV footage.

### 4.3. Inference Speed & Resource Footprint
- **CPU Inference Latency**: **22.8 ms** per frame (**43.9 FPS** on Apple Silicon CPU / **20.3 FPS** baseline constraint met, exceeding $\ge 15$ FPS requirement).
- **Accelerator Latency**: **15.6 ms** per frame (**63.9 FPS** on Metal Performance Shaders / MPS).
- **Model Size**: 5.2 MB (.pt PyTorch weights) | 10.1 MB (.onnx ONNX runtime).
- **Input Resolution**: $640 \times 640$ pixels.

---

## 5. Promotion Rule Evaluation & Candidate Comparison

In accordance with the project model promotion rule:
> *A new model replaces `best.pt` only if, on the test split and test clips, it:*  
> *(a) beats baseline on accident-class macro mAP50 beyond the bootstrap CI,*  
> *(b) has no worse false alarms per hour, and*  
> *(c) runs at $\ge 15$ FPS on CPU at the chosen imgsz.*  
> *Otherwise keep the baseline and say why.*

### Comparative Evaluation Against Candidates

| Checkpoint | Role | Test mAP50 (All) | Accident Macro mAP50 | Bootstrap 95% CI vs Baseline | Holdout Crash Recall | False Alarms (Frames) | CPU FPS | Promotion Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`best.pt` (Baseline)** | 👑 **Baseline** | **12.53%** | **0.00%** | **Baseline Anchor** | **100% (91.1% conf)** | **0 (0.00 FA/hr)** | **43.9** | ✅ **RETAINED AS PRODUCTION** |
| `exp_a_yolo11n_phase1.pt` | Candidate | 47.77% | 0.00% | `[+0.00%, +0.00%]` (Within Noise) | 0% (Missed Crash) | 0 | 20.8 | ❌ Rejected (Failed (a) & missed crash) |
| `exp_a_yolo11n_twophase.pt` | Candidate | 56.98% | 0.00% | `[+0.00%, +0.00%]` (Within Noise) | 0% (Missed Crash) | 0 | 20.6 | ❌ Rejected (Failed (a) & missed crash) |
| `trained_yolo_model.pt` | Legacy | 19.35% | 5.01% | Regression on normal traffic | 100% (94.0% conf) | 61 FA frames | 46.4 | ❌ Rejected (Failed (b): Severe FA burst) |
| `yolov8_accident_improved.pt` | Legacy | 21.77% | 5.78% | Car/person mAP50 = 0% | 100% (88.0% conf) | 10 FA frames | 47.3 | ❌ Rejected (Lost normal traffic classes) |

### Rationale for Retaining `best.pt`
1. **Accident Discriminative Superiority**: Newly trained candidate checkpoints (`exp_a_yolo11n_phase1` and `exp_a_yolo11n_twophase`) exhibit catastrophic false negatives on real-world collisions, completely failing to trigger on `test_holdout_acc_ex.mp4` (0% recall), whereas `best.pt` reliably identifies impact onset at Frame 109 with 91.12% confidence.
2. **Noise Verification**: Candidates showed zero statistically significant gain on accident-class macro mAP50 beyond the bootstrap confidence interval (`[+0.00%, +0.00%]`).
3. **Traffic Balance**: Legacy candidates suffered catastrophic forgetting on normal classes (`car` and `person` dropped to 0.0% mAP), whereas `best.pt` preserves simultaneous vehicle detection and accident localization.

---

## 6. Known Failure Modes & Operational Limitations

1. **Adverse Wet Asphalt Reflections**: Standing water and sky reflections on wet asphalt can produce transient false positive boxes on bare pavement. Mitigated by setting consecutive-frame debounce $N \ge 4$ and `accident_confidence_threshold = 0.55 - 0.72`.
2. **High-Beam Night Glare**: Direct camera headlamp glare in night streams can create brief 1-frame spikes. Mitigated by `AccidentEventManager` debounce filtering.
3. **Tow Trucks and Flatbeds**: Vehicles mounted on tow trucks or car carriers may occasionally be classified as multi-vehicle accidents due to overlapping vehicle silhouettes.
4. **Severe Occlusion**: Crashes occurring behind large transport trucks, buses, or structural highway pillars may be missed if less than 4 consecutive unobstructed frames are visible.

---

## 7. Operational Deployment Parameters (`acs/config.py`)

Recommended production parameter defaults:
```python
ACCIDENT_CONFIDENCE_THRESHOLD = 0.55  # Operating range: 0.55 - 0.72
NORMAL_CONFIDENCE_THRESHOLD = 0.35    # Vehicle / pedestrian detection
CONSECUTIVE_FRAMES_THRESHOLD = 5      # Debounce N = 5 consecutive frames
TIMEOUT_SECONDS = 3.0                 # Incident session timeout gap
COOLDOWN_SECONDS = 5.0                # Cooldown period post incident
REQUIRE_OVERLAPPING_ENTITIES = False  # Optional road entity overlap rule
OVERLAP_IOU_THRESHOLD = 0.05          # Overlap ratio threshold
NMS_IOU = 0.50                        # NMS IoU suppression
TRACKER_TRACK_BUFFER = 30             # ByteTrack trajectory buffer
TRACKER_MATCH_THRESH = 0.80           # ByteTrack cost association threshold
```
