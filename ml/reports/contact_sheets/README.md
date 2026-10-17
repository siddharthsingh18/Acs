# ACS Pre-Label Spot-Check & Eyeball Review Guide
## Automated Video Frame Inference (Baseline `best_v0.pt`)

**Generation Timestamp**: 2026-10-17 04:17:30  
**Confidence Threshold**: `conf >= 0.60`  
**Pre-Labeled Images Generated**: `757` images in `ml/data/pre_labels/`

> [!IMPORTANT]
> **Operational Rule**: Pre-labels in `ml/data/pre_labels/` are strictly quarantined for review and **have NOT been added to the active training set**.
> The validation (`val`) and test (`test`) splits were **NEVER auto-labeled**.

### 1. Pre-Label Detection Summary by Target Class

| Class ID | Class Name | Category | High-Conf Detections (>=0.60) | Review Status | Contact Sheet |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0 | `bike` | Normal | 257 | 🔍 **Ready for Spot-Check** (50 samples) | [`contact_sheet_bike.jpg`](contact_sheet_bike.jpg) |
| 1 | `bike_bike_accident` | Accident | 0 | No high-conf predictions | — |
| 2 | `bike_object_accident` | Accident | 0 | No high-conf predictions | — |
| 3 | `bike_person_accident` | Accident | 0 | No high-conf predictions | — |
| 4 | `car` | Normal | 1,708 | 🔍 **Ready for Spot-Check** (50 samples) | [`contact_sheet_car.jpg`](contact_sheet_car.jpg) |
| 5 | `car_bike_accident` | Accident | 24 | 🔍 **Ready for Spot-Check** (50 samples) | [`contact_sheet_car_bike_accident.jpg`](contact_sheet_car_bike_accident.jpg) |
| 6 | `car_car_accident` | Accident | 232 | 🔍 **Ready for Spot-Check** (50 samples) | [`contact_sheet_car_car_accident.jpg`](contact_sheet_car_car_accident.jpg) |
| 7 | `car_object_accident` | Accident | 53 | 🔍 **Ready for Spot-Check** (50 samples) | [`contact_sheet_car_object_accident.jpg`](contact_sheet_car_object_accident.jpg) |
| 8 | `car_person_accident` | Accident | 0 | No high-conf predictions | — |
| 9 | `person` | Normal | 128 | 🔍 **Ready for Spot-Check** (50 samples) | [`contact_sheet_person.jpg`](contact_sheet_person.jpg) |

---

### 2. Eyeball Inspection Instructions

1. **Review each Contact Sheet**: Open the image file for each class under `ml/reports/contact_sheets/`.
2. **Verification Criteria**:
   - Confirm the bounding box accurately encloses the collision event (not merely unaffected background cars).
   - Verify that collision subtype matches the class definition (e.g., `car_car_accident` shows vehicle impact, `car_object_accident` shows vehicle impact with guardrail/tree/barrier).
3. **Promotion to Training**: Only after visual spot-check approval should high-precision candidate clusters be promoted to `ml/data/merged/images/train/`.
