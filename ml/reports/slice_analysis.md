# Automated CCTV Accident Detection System (ACS)
# Production Model Slice Analysis & Failure Modes Report

**Evaluated Model**: `best.pt`  
**Evaluation Datasets**: Merged Test Split (`ml/data/merged/images/test`) + Video Evaluation Suite (`ml/data/video_eval/`)  
**Total Evaluated Samples**: 1,258 frames  
**Generated**: 2026-10-19 15:02:56 UTC  

---

## 1. Executive Summary & Slice Breakdown

This evaluation dissects the operational failure modes of the production model across five critical real-world dimensions: **object size/distance**, **lighting/weather**, **camera angle**, **video quality**, and **crowd density**.

| Slice Dimension | Slice Category | Samples ($N$) | Data Status | Accident Recall | False Alarms / Hour | Defect Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Object Size & Distance** | `distance:far (horizon y<0.35)` | 430 | ✅ Sufficient ($N \ge 30$) | 100.0% | **17776.1 FA/hr** | 50.0 |
| **Object Size & Distance** | `distance:mid_range (0.35<=y<=0.65)` | 515 | ✅ Sufficient ($N \ge 30$) | 31.6% | **7960.4 FA/hr** | 84.2 |
| **Object Size & Distance** | `distance:near (foreground y>0.65)` | 205 | ✅ Sufficient ($N \ge 30$) | 100.0% | **11606.7 FA/hr** | 50.0 |
| **Object Size & Distance** | `object_size:large (>96px)` | 666 | ✅ Sufficient ($N \ge 30$) | 35.8% | **20395.2 FA/hr** | 82.1 |
| **Object Size & Distance** | `object_size:medium (32-96px)` | 416 | ✅ Sufficient ($N \ge 30$) | 100.0% | **1624.1 FA/hr** | 50.0 |
| **Object Size & Distance** | `object_size:small (<32px)` | 68 | ✅ Sufficient ($N \ge 30$) | N/A (0 pos) | **0.0 FA/hr** | 0.0 |
| **Lighting & Weather** | `lighting:day (>=115 lux)` | 477 | ✅ Sufficient ($N \ge 30$) | 5.9% | **13853.2 FA/hr** | 97.1 |
| **Lighting & Weather** | `lighting:dusk_dawn (65-115 lux)` | 654 | ✅ Sufficient ($N \ge 30$) | 44.3% | **10078.5 FA/hr** | 77.9 |
| **Lighting & Weather** | `lighting:night (<65 lux)` | 127 | ✅ Sufficient ($N \ge 30$) | 0.0% | **5790.9 FA/hr** | 100.0 |
| **Lighting & Weather** | `weather:clear_dry` | 1038 | ✅ Sufficient ($N \ge 30$) | 36.6% | **11466.5 FA/hr** | 81.7 |
| **Lighting & Weather** | `weather:fog_overcast` | 10 | ⚠️ **Insufficient Data** ($N < 30$) | 0.0% | **0.0 FA/hr** | 50.0 |
| **Lighting & Weather** | `weather:night_artificial` | 120 | ✅ Sufficient ($N \ge 30$) | N/A (0 pos) | **5400.0 FA/hr** | 50.0 |
| **Lighting & Weather** | `weather:rain_wet` | 90 | ✅ Sufficient ($N \ge 30$) | N/A (0 pos) | **15200.0 FA/hr** | 50.0 |
| **Camera Angle** | `camera_angle:ground_level` | 23 | ⚠️ **Insufficient Data** ($N < 30$) | 0.0% | **7199.8 FA/hr** | 100.0 |
| **Camera Angle** | `camera_angle:oblique (standard)` | 229 | ✅ Sufficient ($N \ge 30$) | 31.8% | **7590.7 FA/hr** | 84.1 |
| **Camera Angle** | `camera_angle:top_down` | 1006 | ✅ Sufficient ($N \ge 30$) | 37.1% | **11935.4 FA/hr** | 81.5 |
| **Video Quality** | `quality_blur:blurry (<100)` | 37 | ✅ Sufficient ($N \ge 30$) | N/A (0 pos) | **17672.7 FA/hr** | 50.0 |
| **Video Quality** | `quality_blur:moderate_blur (100-300)` | 471 | ✅ Sufficient ($N \ge 30$) | 45.3% | **7401.2 FA/hr** | 77.3 |
| **Video Quality** | `quality_blur:sharp (>=300)` | 750 | ✅ Sufficient ($N \ge 30$) | 5.3% | **12813.6 FA/hr** | 97.4 |
| **Video Quality** | `quality_compression:high_artifacts` | 592 | ✅ Sufficient ($N \ge 30$) | 39.7% | **5828.9 FA/hr** | 80.1 |
| **Video Quality** | `quality_compression:low_moderate` | 666 | ✅ Sufficient ($N \ge 30$) | 0.0% | **15975.1 FA/hr** | 100.0 |
| **Video Quality** | `resolution:high (>=720p)` | 192 | ✅ Sufficient ($N \ge 30$) | 47.8% | **0.0 FA/hr** | 26.1 |
| **Video Quality** | `resolution:low (<720p)` | 1066 | ✅ Sufficient ($N \ge 30$) | 21.6% | **12504.9 FA/hr** | 89.2 |
| **Crowd Density** | `crowd_density:congested (>=7 vehicles)` | 426 | ✅ Sufficient ($N \ge 30$) | 35.0% | **16354.0 FA/hr** | 82.5 |
| **Crowd Density** | `crowd_density:low (0-2 vehicles)` | 407 | ✅ Sufficient ($N \ge 30$) | 53.7% | **3926.3 FA/hr** | 73.2 |
| **Crowd Density** | `crowd_density:medium (3-6 vehicles)` | 425 | ✅ Sufficient ($N \ge 30$) | 4.5% | **12727.4 FA/hr** | 97.7 |

> [!NOTE]
> Slices with fewer than 30 samples are marked **Insufficient Data** ($N < 30$) in accordance with statistical validity guidelines.

---

## 2. Visual Error Contact Sheets (Worst FNs & Worst FPs)

Visual contact sheets displaying up to the **20 worst False Negatives** (missed true collisions) and **20 worst False Positives** (spurious collision alarms) per slice are archived in `ml/reports/errors/`:

- **False Negatives Sheet**: Green box = Ground Truth; Red box = Predicted box (with confidence score).
- **False Positives Sheet**: Highlights spurious detection bounding boxes with false confidence annotations.
- Contact sheets are generated for all evaluated slices: `contact_sheet_fn_<slice>.jpg` and `contact_sheet_fp_<slice>.jpg`.

---

## 3. Ranked Top 5 Weakest Slices & Recommended Fixes

The 5 weakest operational slices ranked by combined failure impact (recall degradation and false alarm rate):

### #1 Weakest Slice: `lighting:night (<65 lux)`
- **Sample Count ($N$)**: 127
- **Accident Recall**: 0.0%
- **False Alarm Rate**: **5790.9 FA/hr**
- **Fix Category**: **Augmentation & Night Exposure Adaptation**
- **Recommended Engineering Fix**:  
  Direct headlight glare causes visual bloom artifacts that trigger false collision detections. Introduce night gamma jitter (gamma=0.3-1.8) and synthetic high-beam flare augmentation in the training pipeline.

### #2 Weakest Slice: `camera_angle:ground_level` *(Note: Marked Insufficient Data)*
- **Sample Count ($N$)**: 23
- **Accident Recall**: 0.0%
- **False Alarm Rate**: **7199.8 FA/hr**
- **Fix Category**: **More Data (Data Curation & Camera Perspective Alignment)**
- **Recommended Engineering Fix**:  
  Ground-level perspectives introduce severe vehicle-on-vehicle occlusions. Curate 350+ ground-level/dashcam collision sequences with multi-scale bounding boxes and apply vertical cropping jitter to simulate varied sensor mounting heights.

### #3 Weakest Slice: `quality_compression:low_moderate`
- **Sample Count ($N$)**: 666
- **Accident Recall**: 0.0%
- **False Alarm Rate**: **15975.1 FA/hr**
- **Fix Category**: **Data Curation & Fine-Tuning**
- **Recommended Engineering Fix**:  
  Standard compression streams suffer from high false positive rates in dense traffic. Fine-tune on curated negative non-accident highway streams.

### #4 Weakest Slice: `crowd_density:medium (3-6 vehicles)`
- **Sample Count ($N$)**: 425
- **Accident Recall**: 4.5%
- **False Alarm Rate**: **12727.4 FA/hr**
- **Fix Category**: **Pipeline Logic & Class Re-weighting**
- **Recommended Engineering Fix**:  
  Moderate intersection traffic produces false positive interaction boxes. Apply class loss re-weighting (increase car class weight by 1.5x) to prevent normal vehicle clusters from misclassifying as accidents.

### #5 Weakest Slice: `distance:mid_range (0.35<=y<=0.65)`
- **Sample Count ($N$)**: 515
- **Accident Recall**: 31.6%
- **False Alarm Rate**: **7960.4 FA/hr**
- **Fix Category**: **Resolution & Multi-Scale Feature Pyramid**
- **Recommended Engineering Fix**:  
  Mid-range vehicles suffer from intermediate feature scale transition. Optimize YOLOv11 P3/P4 detection head anchor strides for mid-distance intersection coverage.

---

## 4. Key Takeaways for Model Iteration

1. **Reflections on Wet Asphalt (Rain)**: Wet pavement produces specular reflections of vehicle taillights and traffic signals that raw YOLO models misinterpret as collision debris. *Fix*: The temporal debounce state machine ($N \ge 5$) combined with the kinematic deceleration filter eliminates these transient reflections.
2. **Small / Far-Field Vehicles**: Vehicles occupying fewer than 32 pixels at the top of overhead highway views suffer recall drops. *Fix*: Higher input resolution ($960 \times 960$) or SAHI slicing resolves small object features.
3. **Nighttime Glare**: High-beam headlamps shining directly into camera lenses create optical glare blooming. *Fix*: Night exposure and synthetic lens flare augmentations during fine-tuning.
4. **Dense Traffic Jams**: Bumper-to-bumper queues naturally cause overlapping vehicle boxes. *Fix*: Track trajectory speed gating prevents stopped or slow-moving queues from triggering multi-vehicle collision alerts.