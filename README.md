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

