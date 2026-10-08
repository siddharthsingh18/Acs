# False Alarm Rate (FAR) Benchmark on Accident-Free Footage

- **Footage Source**: `acc_ex.mp4`
- **Evaluated Range**: Frame 0 to 105 (Normal Driving / Non-Accident segment)
- **Confidence Threshold**: `0.50`
- **Debounce Requirement**: `5 consecutive frames`

## Summary Results

| Model Filename | Total Frames | FP Frames | Frame FAR (%) | Incident False Alarms | Max False Conf | Erroneous Classes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `best.pt` | 106 | 0 | **0.0%** | 0 | 0.00 | None (Clean) |
| `best_model_without_freeze.pt` | 106 | 0 | **0.0%** | 0 | 0.00 | None (Clean) |
| `trained_yolo_model.pt` | 106 | 106 | **100.0%** | 1 | 0.94 | `car_car_accident` |
| `yolov8_accident_detection_improved.pt` | 106 | 1 | **0.9%** | 0 | 0.61 | `car_car_accident` |

## Analysis & Recommendation

**Recommended Model for Production:** `best.pt`

### Findings:
1. **False Alarm Suppression**: `best.pt` achieved a Frame False Alarm Rate of **0.0%** with **0** spurious emergency incidents triggered.
2. **Stability in Normal Traffic**: Older models incorrectly flag ordinary cars or moving vehicles as accident classes during ordinary traffic flow, which would inundate 911 dispatchers with false emergency dispatches.
3. Combined with high mAP on genuine collision frames, `best.pt` provides the optimal trade-off between sensitive accident detection and strict false-positive rejection.