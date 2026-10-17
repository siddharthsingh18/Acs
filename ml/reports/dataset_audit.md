# Automated CCTV Accident Detection System (ACS)
## Multi-Dataset Audit & Label Quality Report

**Execution Date**: 2026-10-18 03:48:29  
**Audit Script**: `ml/scripts/audit_dataset.py`  
**Class Map File**: `ml/configs/class_map.yaml`

This document audits all downloaded raw datasets prior to merging or fine-tuning, strictly evaluating bounding box geometric validity, image integrity, perceptual duplicates (dHash), cross-split leakage, class balance, and semantic label alignment against the 10 target classes.

### 1. Executive Summary Table

| Dataset Name | Total Images | Total Boxes | Empty Labels | Corrupt | Duplicates | Imbalance Ratio | Label Audit Decision | Quality Flags |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `aitc2025_traffic_object_detection` | 2,445 | 24,217 | 14 (0.6%) | 0 | 1 (0.0%) | 53.1:1 | ✅ **Approved** (Normal Traffic) | `NOTICE: Severe class imbalance (ratio 53.1:1).` |
| `hilmantm_traffic_accident_detection` | 1,012 | 3,440 | 0 (0.0%) | 0 | 18 (1.8%) | 4.2:1 | ⚠️ **Partial** (Vehicle mapped, Accident dropped) | `WARNING: Data leakage detected! 6 identical images present across train/val/test splits.` |
| `justjuu_cctv_accident_detection` | 2,763 | 5,400 | 0 (0.0%) | 0 | 692 (25.0%) | 1.3:1 | ❌ **Rejected / Dropped** (Defective labels & duplicates) | `WARNING: High image duplication rate (692/2763 = 25.0% duplicates).`<br>`WARNING: Data leakage detected! 46 identical images present across train/val/test splits.` |
| `on_the_roads_traffic_accident` | 120 | 131 | 0 (0.0%) | 0 | 0 (0.0%) | 1:1 | ❌ **Rejected / Dropped** (Low volume & coarse boxes) | Clean |

---

### 2. Class Mapping Matrix (`ml/configs/class_map.yaml`)

Every source class was mapped to one of the 10 target classes (`bike`, `bike_bike_accident`, `bike_object_accident`, `bike_person_accident`, `car`, `car_bike_accident`, `car_car_accident`, `car_object_accident`, `car_person_accident`, `person`) or explicitly marked as `"drop"` following rigorous empirical inspection of a 20-sample visual audit per generic class:

| Dataset | Source Class ID & Name | Target Class | Action | Empirical Audit Rationale |
| :--- | :--- | :---: | :---: | :--- |
| `aitc2025_traffic_object_detection` | `0: Motorcycle` | `bike` | 🟩 **MAP** | Direct semantic match to 'bike' class (covers motorcycles and bicycles) per ACS guidelines. |
| `aitc2025_traffic_object_detection` | `1: bus` | `car` | 🟩 **MAP** | Normal traffic passenger vehicle mapped to 'car' per normal-traffic rules. |
| `aitc2025_traffic_object_detection` | `2: car` | `car` | 🟩 **MAP** | Direct semantic match to normal traffic 'car'. |
| `aitc2025_traffic_object_detection` | `3: pickup-truck` | `car` | 🟩 **MAP** | Truck variant mapped to normal traffic 'car' per normal-traffic rules. |
| `aitc2025_traffic_object_detection` | `4: trailer` | `car` | 🟩 **MAP** | Heavy motor vehicle mapped to normal traffic 'car' per normal-traffic rules. |
| `aitc2025_traffic_object_detection` | `5: van` | `car` | 🟩 **MAP** | Utility motor vehicle mapped to normal traffic 'car' per normal-traffic rules. |
| `hilmantm_traffic_accident_detection` | `0: accident` | `drop` | 🟥 **DROP** | Empirical 20-sample visual audit revealed a heterogeneous mixture: only ~25% represent car-car collisions; ~15% are car-bike collisions (e.g. accidentFrame1851), ~10% are single-car rollovers/barrier impacts (FLIP_CAR series, car_object_accident), and the remainder are ambiguous aftermath. Per ACS rules, generic accident cannot be mapped to car_car_accident unless purely car-car crashes. Dropped to preserve taxonomy purity.
 |
| `hilmantm_traffic_accident_detection` | `1: vehicle` | `car` | 🟩 **MAP** | Normal-traffic passenger vehicle mapped to 'car' per normal-traffic rules (2,779 valid boxes). |
| `justjuu_cctv_accident_detection` | `0: accident` | `drop` | 🟥 **DROP** | Empirical 20-sample visual audit revealed heterogeneous collisions: car-car, bike-bike collisions (cctv_train_00902), obstacle impacts (cctv_train_02069, cctv_train_00026), and pedestrian incidents. Also exhibits anomalous labelling (e.g. cctv_train_01906 has 9 accident boxes on non-incident traffic). Dropped to avoid severe model degradation.
 |
| `justjuu_cctv_accident_detection` | `1: non_accident` | `drop` | 🟥 **DROP** | Generic negative background class containing cars, buses, bikes, and pedestrians without distinction. Cannot be mapped to 'car' or 'bike' without introducing false labels. Must be dropped.
 |
| `on_the_roads_traffic_accident` | `0: Accident` | `drop` | 🟥 **DROP** | Empirical 20-sample visual audit showed mixed collisions: car-car, car-motorcycle strikes (accident_14), and multiple bike/pedestrian incidents (accident_06, accident_07, accident_09). Many bounding boxes enclose wide scene regions or normal traffic. Dropped.
 |

---

### 3. Detailed Dataset Breakdown

#### Dataset: `aitc2025_traffic_object_detection`

- **Splits Configuration**: train: 2,385 images, val: 30 images, test: 30 images
- **Total Image Count**: 2,445
- **Total Bounding Boxes**: 24,217
- **Duplicate Images (dHash)**: 1 across 1 clusters
- **Cross-Split Leakage Clusters**: 0
- **Invalid / Out-of-Range Boxes**: 0
- **Oversized (>85% area) Canvas Boxes**: 0
- **Visual Samples Exported**: `12` images in `ml/reports/samples/aitc2025_traffic_object_detection/`


**Class Distribution Table**:

| Class ID | Declared Class Name | Box Count | Distribution (%) |
| :---: | :--- | :---: | :---: |
| 0 | `Motorcycle` | 641 | 2.65% |
| 1 | `bus` | 596 | 2.46% |
| 2 | `car` | 18,570 | 76.68% |
| 3 | `pickup-truck` | 3,526 | 14.56% |
| 4 | `trailer` | 350 | 1.45% |
| 5 | `van` | 534 | 2.21% |

**Label Quality Assessment & Flags**:

- NOTICE: Severe class imbalance (ratio 53.1:1).


#### Dataset: `hilmantm_traffic_accident_detection`

- **Splits Configuration**: train: 662 images, val: 295 images, test: 55 images
- **Total Image Count**: 1,012
- **Total Bounding Boxes**: 3,440
- **Duplicate Images (dHash)**: 18 across 16 clusters
- **Cross-Split Leakage Clusters**: 6
- **Invalid / Out-of-Range Boxes**: 0
- **Oversized (>85% area) Canvas Boxes**: 11
- **Visual Samples Exported**: `12` images in `ml/reports/samples/hilmantm_traffic_accident_detection/`
- **Generic Accident Inspection**: `20` images in `ml/reports/samples/hilmantm_traffic_accident_detection/accident_inspection_20/`


**Class Distribution Table**:

| Class ID | Declared Class Name | Box Count | Distribution (%) |
| :---: | :--- | :---: | :---: |
| 0 | `accident` | 661 | 19.22% |
| 1 | `vehicle` | 2,779 | 80.78% |

**Cross-Split Data Leakage Detected (Identical images across splits)**:

```text
Splits: {'val', 'train'} -> Files: ['FLIP_CAR_V141_jpg.rf.281a2dce5e27feadba59bd2769c6036d.jpg', 'FLIP_CAR_V141_jpg.rf.601b4d40f9cbf85114227617cdaf0ead.jpg']
Splits: {'val', 'train'} -> Files: ['Semi-RealisticVD-55_jpg.rf.94de545445f9a7f96f1209ac7703586f.jpg', 'Semi-RealisticVD-55_jpg.rf.6f641d6a48bff7735d76ab269fbf1b1c.jpg']
Splits: {'val', 'train'} -> Files: ['images-19_jpg.rf.e79e1e83a0ff1a8235ca89a188418a71.jpg', 'images-19_jpg.rf.a41bcbf4265a59a26a4fed6676b70505.jpg']
Splits: {'val', 'train'} -> Files: ['images-20_jpg.rf.d162950e5e5f2e227a3b90e70c95f473.jpg', 'download-10_jpg.rf.6f67d7157e14b57cdd29fb79c5fb80fb.jpg']
Splits: {'val', 'train'} -> Files: ['severityV4_38_jpg.rf.27233e587ba1eaae102c4414d88191de.jpg', 'download-6_jpg.rf.40427e5325ab7743937e77ae4eea7caf.jpg', 'severityV4_38_jpg.rf.ecbe61aa0fe1edc4b46f751a186df9b4.jpg']
```

**Label Quality Assessment & Flags**:

- WARNING: Data leakage detected! 6 identical images present across train/val/test splits.


#### Dataset: `justjuu_cctv_accident_detection`

- **Splits Configuration**: train: 2,122 images, val: 316 images, test: 325 images
- **Total Image Count**: 2,763
- **Total Bounding Boxes**: 5,400
- **Duplicate Images (dHash)**: 692 across 473 clusters
- **Cross-Split Leakage Clusters**: 46
- **Invalid / Out-of-Range Boxes**: 0
- **Oversized (>85% area) Canvas Boxes**: 4
- **Visual Samples Exported**: `12` images in `ml/reports/samples/justjuu_cctv_accident_detection/`
- **Generic Accident Inspection**: `20` images in `ml/reports/samples/justjuu_cctv_accident_detection/accident_inspection_20/`


**Class Distribution Table**:

| Class ID | Declared Class Name | Box Count | Distribution (%) |
| :---: | :--- | :---: | :---: |
| 0 | `accident` | 3,042 | 56.33% |
| 1 | `non_accident` | 2,358 | 43.67% |

**Cross-Split Data Leakage Detected (Identical images across splits)**:

```text
Splits: {'val', 'train'} -> Files: ['cctv_train_00037.jpg', 'cctv_train_01869.jpg', 'cctv_val_00021.jpg', 'cctv_val_00044.jpg', 'cctv_val_00233.jpg']
Splits: {'val', 'test', 'train'} -> Files: ['cctv_train_00056.jpg', 'cctv_train_00493.jpg', 'cctv_train_00743.jpg', 'cctv_train_00812.jpg', 'cctv_train_01302.jpg', 'cctv_train_01470.jpg', 'cctv_train_01560.jpg', 'cctv_train_01650.jpg', 'cctv_train_01876.jpg', 'cctv_val_00025.jpg', 'cctv_val_00065.jpg', 'cctv_val_00188.jpg', 'cctv_test_00023.jpg', 'cctv_test_00044.jpg', 'cctv_test_00098.jpg', 'cctv_test_00136.jpg']
Splits: {'test', 'train'} -> Files: ['cctv_train_00068.jpg', 'cctv_train_00078.jpg', 'cctv_train_00433.jpg', 'cctv_train_01042.jpg', 'cctv_train_01422.jpg', 'cctv_train_01598.jpg', 'cctv_train_01769.jpg', 'cctv_train_01855.jpg', 'cctv_train_01883.jpg', 'cctv_train_01932.jpg', 'cctv_train_01972.jpg', 'cctv_test_00051.jpg']
Splits: {'val', 'train'} -> Files: ['cctv_train_00069.jpg', 'cctv_train_00112.jpg', 'cctv_train_00994.jpg', 'cctv_train_01088.jpg', 'cctv_train_01365.jpg', 'cctv_train_01582.jpg', 'cctv_train_01671.jpg', 'cctv_train_01790.jpg', 'cctv_train_01891.jpg', 'cctv_train_01954.jpg', 'cctv_val_00298.jpg']
Splits: {'val', 'test', 'train'} -> Files: ['cctv_train_00072.jpg', 'cctv_train_01329.jpg', 'cctv_val_00181.jpg', 'cctv_test_00261.jpg']
```

**Label Quality Assessment & Flags**:

- WARNING: High image duplication rate (692/2763 = 25.0% duplicates).
- WARNING: Data leakage detected! 46 identical images present across train/val/test splits.


#### Dataset: `on_the_roads_traffic_accident`

- **Splits Configuration**: train: 105 images, val: 10 images, test: 5 images
- **Total Image Count**: 120
- **Total Bounding Boxes**: 131
- **Duplicate Images (dHash)**: 0 across 0 clusters
- **Cross-Split Leakage Clusters**: 0
- **Invalid / Out-of-Range Boxes**: 0
- **Oversized (>85% area) Canvas Boxes**: 0
- **Visual Samples Exported**: `12` images in `ml/reports/samples/on_the_roads_traffic_accident/`
- **Generic Accident Inspection**: `20` images in `ml/reports/samples/on_the_roads_traffic_accident/accident_inspection_20/`


**Class Distribution Table**:

| Class ID | Declared Class Name | Box Count | Distribution (%) |
| :---: | :--- | :---: | :---: |
| 0 | `Accident` | 131 | 100.00% |

**Label Quality Assessment & Flags**:

- PASS: No severe defects detected.


---

### 4. Visual Sampling & Eyeball Inspection Guide

Twelve (12) high-resolution annotated sample images per dataset have been rendered with bounding boxes and class headers into `ml/reports/samples/<dataset_folder>/`:

- **`aitc2025_traffic_object_detection`**: `ml/reports/samples/aitc2025_traffic_object_detection/sample_*.jpg`
- **`hilmantm_traffic_accident_detection`**: `ml/reports/samples/hilmantm_traffic_accident_detection/sample_*.jpg`
  - Generic Accident Focus: `ml/reports/samples/hilmantm_traffic_accident_detection/accident_inspection_20/accident_*.jpg`
- **`justjuu_cctv_accident_detection`**: `ml/reports/samples/justjuu_cctv_accident_detection/sample_*.jpg`
  - Generic Accident Focus: `ml/reports/samples/justjuu_cctv_accident_detection/accident_inspection_20/accident_*.jpg`
- **`on_the_roads_traffic_accident`**: `ml/reports/samples/on_the_roads_traffic_accident/sample_*.jpg`
  - Generic Accident Focus: `ml/reports/samples/on_the_roads_traffic_accident/accident_inspection_20/accident_*.jpg`

---

### 5. Final Recommendations & Guardrails (Do Not Merge Yet)

1. **Do Not Merge Raw Datasets Directly**: `justjuu_cctv_accident_detection` has a 25.0% perceptual duplicate rate (692 duplicates) and 46 cross-split data leakage cases. Direct merging would contaminate validation splits and degrade model generalization.
2. **Preserve Fine-Grained Accident Classes**: Generic `accident` labels in `hilmantm`, `justjuu`, and `on_the_roads` must remain mapped to `drop` or undergo automated re-annotation using the fine-grained model, as they mix rollovers, bike-bike, car-bike, and car-car collisions indiscriminately.
3. **High-Value Negative Training**: `aitc2025_traffic_object_detection` provides 24,217 pristine normal vehicle bounding boxes (`Motorcycle` -> `bike`, others -> `car`) which will substantially reduce false-alarm accident detections in heavy normal traffic.
4. **Flag obviously bad labels**: Datasets with severe data leakage, high duplicate rates, or unverified generic labels are rejected or quarantined as candidates pending rigorous cleaning.
