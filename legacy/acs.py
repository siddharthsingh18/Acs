import cv2
import streamlit as st
import pandas as pd
import os
import time
from config import config, load_yolo_model, ACCIDENT_CLASSES

# Load the primary YOLO model from configuration
model = load_yolo_model(config.model_path)

# Ensure necessary output directories exist
config.ensure_directories()

# Initialize the session state for accident reports if it doesn't exist
if 'report_filenames' not in st.session_state:
    st.session_state.report_filenames = []

# Function to generate and save an accident report as CSV
def generate_report(class_name, confidence, image_filename):
    report_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())  # Current time for filename
    report_filename = f"report_{report_time}.csv"
    report_path = os.path.join(config.reports_dir, report_filename)

    # Create a DataFrame for the report
    report_df = pd.DataFrame([{'Class': class_name, 'Confidence': confidence, 'Time': report_time}])

    # Save the DataFrame to a CSV file
    report_df.to_csv(report_path, index=False)

    # Add the report filename and associated image filename to the session state
    st.session_state.report_filenames.append((report_filename, image_filename))
    return report_filename

# Function to process video and display annotated frames
def display_annotated_video(video_path, fps_reduction=4):
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return
    
    # Get video properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    new_fps = max(1, original_fps // fps_reduction)  # Reduce FPS
    frame_time = 1.0 / new_fps  # Time to display each frame

    # Create a placeholder for the video frames
    frame_placeholder = st.empty()
    last_accident_time = 0.0  # Track timestamp of the last detected accident for cooldown

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Annotate frame using configured confidence and device
        results = model.predict(source=frame, conf=config.confidence_threshold, device=config.device)
        annotated_frame = results[0].plot()  # Annotated frame

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the annotated frame in the placeholder
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        boxes = results[0].boxes  # Access detection boxes

        # Check for accidents using ACCIDENT_CLASSES (car and person never trigger alerts)
        for box in boxes:
            class_id = int(box.cls[0])  # Class ID of the detected object
            confidence = float(box.conf[0])  # Confidence score
            class_name = model.names[class_id]  # Get class name

            now = time.time()
            if class_name.lower() in ACCIDENT_CLASSES and (now - last_accident_time >= config.cooldown_seconds):
                last_accident_time = now
                image_filename = f"image_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                image_path = os.path.join(config.report_images_dir, image_filename)
                cv2.imwrite(image_path, frame)  # Save the image

                report_filename = generate_report(class_name, confidence, image_filename)
                st.write(f"Accident Detected: {class_name}, Confidence: {confidence:.2f}. Report saved: {report_filename}")

    cap.release()
    st.write("Video playback complete.")

# Function to show reports
def show_reports():
    st.title("Accident Reports")

    # Sidebar to show report filenames and allow clicking to display report details
    st.header("Accident Reports")
    if st.session_state.report_filenames:
        for report_filename, image_filename in st.session_state.report_filenames:
            if st.button(report_filename):
                # Load and display the report details when clicked
                report_path = os.path.join(config.reports_dir, report_filename)
                report_df = pd.read_csv(report_path)
                st.write(f"Details of {report_filename}:")
                st.dataframe(report_df)

                # Display the associated image
                image_path = os.path.join(config.report_images_dir, image_filename)
                st.image(image_path, caption=f"Image for {report_filename}", use_column_width=True)
    else:
        st.write("No accident reports yet.")

# Streamlit app UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Video & Prediction", "Reports"])

if page == "Video & Prediction":
    st.title("YOLO Annotated Video Display")

    # File uploader for video input
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov","gif"])

    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        video_path = os.path.join(config.temp_dir, 'uploaded_video.mp4')
        with open(video_path, 'wb') as f:
            f.write(uploaded_video.read())
        
        st.write(f"Uploaded video saved to: {video_path}")

        # Display the annotated video using the placeholder 
        display_annotated_video(video_path)

elif page == "Reports":
    show_reports()
