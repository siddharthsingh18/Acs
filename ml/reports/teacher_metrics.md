# Teacher Model Fine-Tuning Metrics (YOLO11m)

> **UNVERIFIED:** The per-class metrics and split counts cannot be reproduced because the teacher checkpoint, dataset, and raw validation artifacts are absent; the report also uses the legacy 10-class taxonomy.

- **Evaluated Model**: `YOLO11m (Teacher)` (20,060,718 parameters)
- **Checkpoint Path**: `models/teacher_yolo11m.pt`
- **Dataset Evaluated**: `ml/data/merged/data.yaml` (FROZEN TEST split, 417 images)
- **Base Architecture**: YOLOv11-Medium with Frozen Backbone Neck Fine-Tuning

### Overall Performance Matrix

| Metric Scope | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **All Classes** | 23.40% | 49.14% | 15.82% | 5.67% |
| **Accident Classes Only** | 0.00% | 0.00% | 0.00% | 0.00% |

### Per-Class Detailed Breakdown

| Class ID | Class Name | Type | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `bike` | 🚗 Normal | 34.98% | 17.05% | 15.87% | 4.46% |
| 1 | `bike_bike_accident` | 🚨 Accident | 0.00% | 0.00% | 0.00% | 0.00% |
| 2 | `bike_object_accident` | 🚨 Accident | 0.00% | 0.00% | 0.00% | 0.00% |
| 3 | `bike_person_accident` | 🚨 Accident | 0.00% | 0.00% | 0.00% | 0.00% |
| 4 | `car` | 🚗 Normal | 11.81% | 81.24% | 15.77% | 6.88% |
| 5 | `car_bike_accident` | 🚨 Accident | 0.00% | 0.00% | 0.00% | 0.00% |
| 6 | `car_car_accident` | 🚨 Accident | 0.00% | 0.00% | 0.00% | 0.00% |
| 7 | `car_object_accident` | 🚨 Accident | 0.00% | 0.00% | 0.00% | 0.00% |
| 8 | `car_person_accident` | 🚨 Accident | 0.00% | 0.00% | 0.00% | 0.00% |
| 9 | `person` | 🚗 Normal | 0.00% | 0.00% | 0.00% | 0.00% |
