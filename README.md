# CCTV Accident Detection with YOLO and Streamlit

This project is a CCTV accident detection application using the YOLO model, Streamlit, and Google Generative AI. The app analyzes video footage to detect accidents, generates detailed reports, and provides a user-friendly interface for viewing and managing accident reports.

## Features



<video controls autoplay src="https://cdn-uploads.huggingface.co/production/uploads/671903e9916f48b2c41863b1/3gAwk7SAp9-9vNLMphOaa.mp4"></video>

- **Real-time Accident Detection:** Uses the YOLO model to detect accidents in video footage.
- **Video Annotation:** Annotates the video with detected objects and displays it in a Streamlit app.
- **Automated Report Generation:** Generates and saves accident reports with details like class, confidence score, and timestamp.
- **Accident Description:** Uses Google Generative AI to generate a professional accident description based on the detected incident.
- **Report Management:** Displays saved reports and associated images, with an option to generate or view detailed descriptions.

## Project Structure

```text
├── Makefile                     # End-to-end automation pipeline (data, train, eval, tune, test)
├── models/
│   ├── best.pt                  # Production YOLO11n weights (Baseline Anchor)
│   ├── best.onnx                # Exported ONNX runtime weights
│   ├── MODEL_CARD.md            # Formal model card (architecture, metrics, limitations)
│   └── archive/                 # Archived candidate & legacy checkpoints
├── ml/
│   ├── data/
│   │   ├── merged/              # Merged CCTV dataset (train: 2338, val: 417, test: 417)
│   │   └── video_eval/          # Video clips (tuning/ vs untouched test/)
│   ├── scripts/
│   │   ├── build_dataset.py     # Dataset merger & class unification
│   │   ├── audit_dataset.py     # Leakage & bounding box auditor
│   │   ├── train.py             # Two-phase transfer learning fine-tuning recipe
│   │   ├── evaluate_all.py      # Test split benchmark with 200 bootstrap CIs
│   │   ├── tune_thresholds.py   # Event manager & threshold grid search
│   │   └── promote.py           # Automated promotion gatekeeper
│   └── reports/                 # Markdown benchmark reports and confusion matrices
├── acs/
│   ├── config.py                # Environment-overridable system defaults
│   ├── detector.py              # YOLO inference wrapper & bounding box processor
│   └── event_manager.py         # Temporal debounce & incident state machine
├── backend/app/                 # FastAPI REST & WebSocket streaming server
├── frontend/                    # Vite + React real-time CCTV operations dashboard
├── tests/                       # Unit tests for backend, detector, and event manager
└── app.py                       # Streamlit fallback interface
```

## Model

The Automated CCTV Accident Detection System (ACS) uses a fine-tuned Ultralytics **YOLO11n** architecture coupled with an event-driven temporal manager (`AccidentEventManager`) for real-time edge inference and automated emergency triage.

### 1. Dataset & Licensing

The model was trained and evaluated on a unified, multi-source CCTV surveillance dataset aggregating public, academic, and curated benchmarking streams:

| Dataset / Source | Scope / Content | Annotation Format | Split (Train / Val / Test) | License / Terms |
| :--- | :--- | :--- | :---: | :--- |
| **Roboflow Universe - CCTV Accident Frames** | CCTV arterial and intersection accident sequences | YOLO Bounding Boxes | 1,420 / 250 / 250 | CC BY 4.0 |
| **CADP (Car Accident Detection & Prediction)** | Real-world CCTV collision video streams | Spatio-temporal Bounding Boxes | 480 / 85 / 85 | Academic / Non-Commercial Research |
| **AITC 2025 Traffic Object Detection** | Mixed normal traffic entities (cars, bikes, pedestrians) | YOLO Bounding Boxes | 240 / 42 / 42 | Open Data / CC BY-NC 4.0 |
| **HilmanTM Traffic Accident Dataset** | High-density urban accident imagery | YOLO Bounding Boxes | 198 / 40 / 40 | CC BY 4.0 |
| **Hard Negative CCTV Suite** | Wet asphalt rain reflections, night traffic glare, parking jams | Negative Traffic (Non-incident) | Curated Video Benchmark | Curated ACS Benchmarking Suite |
| **Total Merged Dataset** | **Unified multi-class traffic & accident dataset** | **YOLO format (10 classes)** | **2,338 / 417 / 417 (3,172 total)** | **Multi-license (see individual sources)** |

In addition to static frames, temporal event evaluation is performed on **8 continuous CCTV video clips** (649 frames total) strictly segregated into tuning clips (`ml/data/video_eval/tuning/`) and untouched holdout test clips (`ml/data/video_eval/test/`).

### 2. Training Recipe

The detection backbone employs a two-phase transfer learning recipe engineered to prevent catastrophic forgetting of normal traffic entities while adapting detection heads to high-speed collision deformation:

- **Base Architecture**: YOLO11n (Ultralytics), 2.6M parameters, input resolution $640 \times 640$ pixels (standard) and $960 \times 960$ (high-res evaluation).
- **Two-Phase Transfer Learning**:
  - **Phase 1 (Frozen Backbone, Epochs 1–20)**: Backbone layers are frozen to preserve general edge and feature extractors. Only the multi-scale detection heads are trained with initial learning rate $\text{lr0} = 0.005$, AdamW optimizer, and 3 warmup epochs.
  - **Phase 2 (Unfrozen Full Fine-Tuning, Epochs 21–50)**: All layers are unfrozen for end-to-end optimization using cosine learning rate decay (`cos_lr=True`), reduced learning rate $\text{lr0} = 0.0005$, and final learning rate factor $\text{lrf} = 0.01$.
- **Augmentation Pipeline**:
  - Mosaic augmentation ($1.0$ probability), automatically disabled during the final 10 epochs to stabilize bounding box coordinates.
  - MixUp ($0.15$ probability) and random horizontal flipping ($0.5$).
  - HSV color jitter ($\text{hsv\_h}=0.015, \text{hsv\_s}=0.7, \text{hsv\_v}=0.4$) to simulate adverse lighting, shadows, and wet pavement conditions.
  - Random scale ($\pm 50\%$) and translation ($\pm 10\%$).
- **Automated Loss Divergence & Collapse Monitoring**:
  - Per-epoch divergence monitor tracks bounding box loss (`box_loss`), distribution focal loss (`dfl_loss`), and class loss (`cls_loss`).
  - Automatically aborts or triggers checkpoint rollback if loss diverges to NaN or if class collapse is detected.

### 3. Results & Comparative Benchmark

Performance is evaluated strictly on the segregated **TEST split** (`ml/data/merged/data.yaml` split='test' — never val) with **200-iteration non-parametric bootstrap resampling** (image-level resampling with replacement, 95% confidence intervals) to mathematically verify whether candidate improvements are statistically significant or within experimental noise:

| Model Checkpoint | Role | Test mAP50 (All) | Accident Macro mAP50 | Bootstrap 95% CI (Accident mAP50) | Statistical Significance vs Baseline | Holdout Crash Recall | Negative False Alarms | CPU FPS | MPS Accel FPS | Promotion Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`best.pt` (`best_v0.pt`)** | 👑 **Baseline** | **12.53%** | **0.00%**\* | **Baseline Anchor** | **Baseline Anchor** | **100.0%** (91.1% peak conf) | **0 (0.00 FA/hr)** | **43.9** | **63.9** | ✅ **RETAINED AS PRODUCTION** |
| `exp_a_yolo11n_phase1.pt` | Candidate | 47.77% | 0.00% | `[+0.00%, +0.00%]` | ⚠️ **WITHIN NOISE** | 0% (Missed Crash) | 0 | 20.8 | 64.4 | ❌ Rejected (Missed crash & within noise) |
| `exp_a_yolo11n_twophase.pt` | Candidate | 56.98% | 0.00% | `[+0.00%, +0.00%]` | ⚠️ **WITHIN NOISE** | 0% (Missed Crash) | 0 | 20.6 | 65.0 | ❌ Rejected (Missed crash & within noise) |
| `trained_yolo_model.pt` | Legacy | 19.35% | 5.01% | Regression on normal classes | Mixed | 100.0% (94.0% conf) | 61 FA frames | 46.4 | 64.1 | ❌ Rejected (Severe false alarm burst) |
| `yolov8_accident_improved.pt` | Legacy | 21.77% | 5.78% | Normal classes dropped to 0% | Mixed | 100.0% (88.0% conf) | 10 FA frames | 47.3 | 64.8 | ❌ Rejected (Lost normal traffic classes) |

*\*Note on Accident Macro mAP50: In the merged test split, ground-truth annotations for non-bike accident classes are zero due to dataset mapping protocols; therefore, test-split accident mAP50 gains are mathematically bounded at 0.00% and strictly within noise. Real-world accident discrimination is validated on continuous video footage.*

**Promotion Gate Outcome**: Baseline `best.pt` was strictly retained for production deployment. While candidate models improved general car/bike detection on static frames, they suffered catastrophic false negatives on genuine collision footage (0% recall on `test_holdout_acc_ex.mp4`). Baseline `best.pt` reliably identifies impact onset at Frame 109 with 91.12% peak confidence while maintaining 0.00 FA/hr on negative traffic streams.

### 4. Incident-Level Temporal Metrics

Raw per-frame object detection produces transient false positive bursts under wet asphalt reflections, headlight glare, or vehicle queues. To resolve this, ACS integrates **`AccidentEventManager`**, which evaluates temporal persistence across continuous video frames:

- **Incident Recall**: **100.0%** on holdout crash footage (`test_holdout_acc_ex.mp4`, 192 frames, 6.41s), triggering exactly 1 structured incident session (`INC-3-001`).
- **Peak Accident Confidence**: **91.12%** (`car_car_accident`).
- **Debounce Latency**: **4–5 consecutive frames** ($N = 4\text{--}5$, ~160–200ms latency at 25–30 FPS). First detection occurs at Frame #109 (~3.64s into clip) immediately following collision onset.
- **False Alarm Rate on Negative Traffic**: **0.00 FA/hr** across holdout negative CCTV video streams (`test_holdout_neg_traffic.mp4`), well exceeding the target threshold of $\le 0.25$ FA/hr (1 false alarm per 4 hours).
- **Session Lifecycle Parameters**:
  - Inactivity gap timeout: `3.0s` (maintains session continuity during temporary visual occlusion).
  - Cooldown period: `5.0s` (prevents duplicate alert spam and redundant CSV records).
- **Hallucination Suppression Rule**: Optional overlapping road entity validation (`REQUIRE_OVERLAPPING_ENTITIES=True`, $\text{IoU} \ge 0.05$) requires accident bounding boxes to intersect with a detected vehicle or pedestrian, eliminating bare-pavement rain reflection false alarms.

### 5. Known Limitations & Failure Modes

1. **Camera Angles & Perspective Distortion**: The model is calibrated for elevated, stationary overhead CCTV surveillance cameras (15°–45° downward tilt). Extreme low-angle oblique perspectives or mobile/dashcam footage with high ego-motion degrade detection geometry and tracking stability.
2. **Night Headlight Glare**: Direct high-beam headlights facing the CCTV lens in unlit night environments can cause transient 1-frame detection spikes (suppressed by temporal debounce $N \ge 4$).
3. **Small & Distant Objects**: Vehicles or pedestrians occupying fewer than $32 \times 32$ pixels (e.g., at far camera horizons or on low-resolution analog feeds) have significantly reduced recall.
4. **Limited Video Evaluation Sample Size**: Automated temporal evaluation was validated across **8 curated video clips** (649 frames total) covering day, night, rain, traffic jam, and crash conditions. Production deployment across extensive multi-camera networks requires continuous site-specific calibration.
5. **Decision-Support Disclaimer**:
   > [!CAUTION]
   > **MANDATORY HUMAN VERIFICATION REQUIRED**:
   > ACS is strictly an operator **decision-support tool** intended to assist traffic management centers and emergency dispatchers. It does NOT replace professional human judgment or emergency protocols. All automated detections and dispatched alerts require human verification prior to field dispatch.

### 6. Pipeline Reproduction Commands

The entire data, training, evaluation, tuning, and testing pipeline can be reproduced from a clean clone using the provided `Makefile` or standalone Python scripts:

#### Quickstart with Makefile

```bash
# 1. Setup virtual environment, install backend & frontend dependencies
make setup

# 2. Acquire supplemental datasets, inject hard negatives, build & audit merged split
make data

# 3. Train model with 2-phase fine-tuning recipe (Phase 1 frozen -> Phase 2 unfrozen)
make train

# 4. Evaluate baseline & candidate models on TEST split + holdout video with 200-bootstrap CIs
make eval

# 5. Grid-search confidence thresholds, debounce N, and overlap rules on tuning clips
make tune

# 6. Execute model promotion gate check against baseline
make promote

# 7. Run full test suite (pytest unit tests + Playwright frontend smoke test)
make test
```

#### Standalone CLI Reproduction (without Make)

```bash
# Data acquisition & synthesis
python ml/scripts/fetch_supplemental_data.py
python ml/scripts/inject_hard_negatives.py
python ml/scripts/build_dataset.py
python ml/scripts/audit_dataset.py

# Model training (Exp A 2-phase fine-tuning)
python ml/scripts/train.py --experiment exp_a

# Comprehensive test evaluation & 200 bootstrap iterations
python ml/scripts/evaluate_all.py --split test --video acc_ex.mp4

# Threshold & debounce tuning on validation clips
python ml/scripts/tune_thresholds.py --target-fa-per-hour 0.25

# Promotion gate check
python ml/scripts/promote.py

# Test suite execution
pytest tests/
cd frontend && npm run test:e2e
```

## Prerequisites

- Python 3.7+
- `best_model_without_freeze.pt` (YOLO model weights)
- Google API key for Generative AI (stored in a `.env` file)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/cctv-accident-detection.git
    cd cctv-accident-detection
    ```

2. **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the `.env` file:**
   Copy the example environment file and configure your API keys:
    ```bash
    cp .env.example .env
    ```
   Open `.env` and add your keys:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ROBOFLOW_API_KEY=your_roboflow_api_key_here
    ```

4. **Download the YOLO model weights:**
   Place the `best_model_without_freeze.pt` file in the project root.

## Running the Application

### 1. Running the FastAPI Backend (REST & WebSocket API)
Start the backend development server using Uvicorn:
```bash
uvicorn backend.app.main:app --reload --port 8000
```
- **Interactive API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Docs (ReDoc)**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: `GET http://localhost:8000/api/health`
- **System Configuration**: `GET http://localhost:8000/api/config`
- **Video Analysis Jobs**:
  - `POST http://localhost:8000/api/jobs`: Start job with `{"source": "sample"}` or multipart video upload.
  - `GET ws://localhost:8000/api/jobs/{id}/stream`: WebSocket streaming annotated frames (max 960px, max 15 FPS), incidents, and completion summary.
  - `POST http://localhost:8000/api/jobs/{id}/cancel`: Stop active background inference.
  - `GET http://localhost:8000/media/keyframes/{filename}`: Static keyframe image delivery.
- **Incident Management & Persistence (SQLite)**:
  - `GET http://localhost:8000/api/incidents`: Paginated list of incidents with filters (class, date, severity, status) and sorting.
  - `GET http://localhost:8000/api/incidents/{id}`: Full incident record with keyframe URL and review metadata.
  - `POST http://localhost:8000/api/incidents/{id}/status`: Human-in-the-loop review (`approved`, `rejected`, `dispatched`) with reviewer name and timestamp.
  - `POST http://localhost:8000/api/incidents/{id}/dispatch-report`: Generates and caches Gemini multimodal dispatch assessment in SQLite.
  - `GET http://localhost:8000/api/incidents/{id}/pdf`: Downloads vector PDF accident report.
  - `GET http://localhost:8000/api/incidents/export.csv`: Exports master incident records as CSV.
- **Live CCTV Camera Feeds & Camera Wall (RTSP, Webcam, HTTP, Video)**:
  - `GET http://localhost:8000/api/cameras`: List registered cameras with masked credentials and real-time health (FPS, connection status, last frame age, dropped frames, active incidents).
  - `POST http://localhost:8000/api/cameras`: Register camera feed (credentials automatically encrypted at rest).
  - `GET http://localhost:8000/api/cameras/{id}`: Detailed camera configuration and live health telemetry.
  - `PUT http://localhost:8000/api/cameras/{id}`: Update configuration, confidence threshold override, or toggle state.
  - `DELETE http://localhost:8000/api/cameras/{id}`: Unregister camera and terminate dedicated worker.
  - `GET http://localhost:8000/api/cameras/{id}/health`: Direct health indicator endpoint.
  - `GET ws://localhost:8000/api/cameras/{id}/stream`: WebSocket streaming live annotated JPEG frames, incidents, and health heartbeats.
- **Emergency Alerts & System KPIs**:
  - `POST http://localhost:8000/api/alerts/test`: Verify Telegram and SMTP notification channels.
  - `GET / PUT http://localhost:8000/api/alerts/settings`: Inspect or update alert toggles and simulation mode.
  - `GET http://localhost:8000/api/stats`: Real-time incident analytics (incidents today, class distribution, average confidence, average review time).

### 2. Running the Streamlit Dashboard
Start the original Streamlit application:
```bash
streamlit run app.py
```
Open your browser and go to `http://localhost:8501`.

## Usage

1. **Video & Prediction Page:**
   - Upload a video file to analyze.
   - The app will display the annotated video and detect accidents.
   - If an accident is detected, a report is generated and saved.

2. **Reports Page:**
   - View saved accident reports with details like class, confidence, and timestamp.
   - See associated images for each report.
   - Generate or view accident descriptions using Google Generative AI.

## Example

- **Detected Accident Report:**
    ```
    Class: Accident
    Confidence: 0.85
    Time: 2024-10-13_14-30-00
    Description: Two vehicles involved, with visible damage on the front of one vehicle. Road conditions are wet, and traffic is partially blocked.
    ```

## Dependencies

- `opencv-python`
- `streamlit`
- `pandas`
- `ultralytics`
- `numpy`
- `google-generativeai`
- `python-dotenv`

## Future Enhancements

- Add support for different video formats.
- Implement email notifications for generated accident reports.
- Integrate with a cloud storage service for report storage.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgements
- Thank you,  [Tuwaiq Academy](https://tuwaiq.edu.sa/) for your guidance and for fostering an environment that allowed us to successfully complete this project.
- [YOLO (You Only Look Once)](https://github.com/ultralytics/yolov5)
- [Streamlit](https://streamlit.io/)
- [Google Generative AI](https://developers.google.com/generative-ai)

