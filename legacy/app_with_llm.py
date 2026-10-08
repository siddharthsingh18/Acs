import cv2
import streamlit as st
import pandas as pd
import os
import time
import numpy as np 
from dotenv import load_dotenv
import random

try:
    from google import genai
except ImportError:
    try:
        import google.generativeai as genai
    except ImportError:
        genai = None

from config import config, load_yolo_model, ACCIDENT_CLASSES

# Ensure environment variables are loaded
load_dotenv()

# Load primary and secondary models from configuration
model = load_yolo_model(config.model_path)
modelf = load_yolo_model(config.injury_model_path) if os.path.exists(config.injury_model_path) else model

# Ensure required directories exist
config.ensure_directories()

# Initialize the session state for accident reports if it doesn't exist
if 'report_filenames' not in st.session_state:
    st.session_state.report_filenames = []

# Function to generate and save an accident report as CSV
def generate_report(class_name,  image_filename, description=None):
    report_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())  # Current time for filename
    report_filename = f"report_{report_time}.csv"
    report_path = os.path.join(config.reports_dir, report_filename)

    # Create a DataFrame for the report, including a Description column
    report_data = {
        'Class': class_name,
        'Time': report_time,
        'Description': description  # Description field
    }
    
    report_df = pd.DataFrame([report_data])

    # Save the DataFrame to a CSV file
    report_df.to_csv(report_path, index=False)

    # Add the report filename and associated image filename to the session state
    st.session_state.report_filenames.append((report_filename, image_filename))
    return report_filename


# Function to generate an accident description based on the image using Google Gemini
def generate_accident_report(image_path, yolo_report=None, time=None):
    # Set up the API key
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return "Failed to generate report: GOOGLE_API_KEY is not set. Please define it in your .env file."
    if genai is None:
        return "Failed to generate report: Google GenAI SDK is not installed. Please install 'google-genai'."

    st.write(yolo_report)
    # Construct the prompt
    prompt = "Describe the accident based on the given image. make proficinal report without mentioning what you are doing \
        make percise and consise so the officer can make decision be very sure whene makegin this report don't mention \
            that this report is based on the image or it may not be accurate "
    if yolo_report:
        prompt += f" this is the calsses names for yolo report: {yolo_report}. Provide more details about the scene."
    if time:
        prompt += f" This incident occurred at {time}."

    # Add specific instructions for the model to make the report useful for 911.
    prompt += (" Include details such as the type and number of vehicles involved and there types and colors, "
               "visible damage, any hazards, road conditions, presence of people, "
               "and any blocked traffic. Provide suggestions for emergency response actions."
               "i need it very short not more then 5 lines"
               "and use spsific template like car1:\n car2:\n short descriptoin:\n and so on add any nedded failds")

    # Choose the model and generate the response using configured gemini_model_name
    try:
        if hasattr(genai, 'Client'):
            # Current google-genai SDK
            client = genai.Client(api_key=api_key)
            uploaded_file = client.files.upload(file=image_path[-1])
            response = client.models.generate_content(
                model=config.gemini_model_name,
                contents=[uploaded_file, prompt]
            )
            return response.text
        else:
            # Fallback for legacy google-generativeai
            genai.configure(api_key=api_key)
            uploaded_file = genai.upload_file(path=image_path[-1], display_name="Accident Scene")
            legacy_model = genai.GenerativeModel(model_name=config.gemini_model_name)
            response = legacy_model.generate_content([uploaded_file, prompt])
            return response.text
    except Exception as e:
        return f"Failed to generate report: {str(e)}"

def display_annotated_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return

    # Create a placeholder for the video frames
    frame_placeholder = st.empty()
    st.session_state.detected_class_names = set()
    list_of_images = []
    class_names = set()
    accident_detectd = False
    last_accident_time = 0.0  # Track timestamp of the last detected accident for cooldown

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Annotate frame
        height, width = frame.shape[:2]
        new_width = 1080
        new_height = int(height * (new_width / width))
        frame_small = cv2.resize(frame, (new_width, new_height))

        # Prediction using configured confidence and device
        results = model.predict(source=frame_small, conf=config.confidence_threshold, device=config.device, imgsz=640)
        
        annotated_frame = results[0].plot()
        boxes_results = results[0].boxes
        all_boxes = list(boxes_results)
        boxes_resultsf = None
        
        if accident_detectd:
            resultsf = modelf.predict(source=annotated_frame, conf=config.injury_confidence_threshold, device=config.device, imgsz=640)
            annotated_frame = resultsf[0].plot() 
            boxes_resultsf = resultsf[0].boxes
            all_boxes = all_boxes + list(boxes_resultsf)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
        else:
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

        if boxes_resultsf:
            for box in boxes_resultsf:
                class_id = int(box.cls[0])  # Class ID of the detected object
                class_name = modelf.names[class_id]
                if class_name not in class_names:
                    st.info(f'{class_name} is detected', icon="ℹ️")
                    class_names.add(class_name)
                class_names.add(class_name)

        for box in boxes_results:
            class_id = int(box.cls[0])  # Class ID of the detected object
            class_name = model.names[class_id]
            if class_name not in class_names:
                st.info(f'{class_name} is detected', icon="ℹ️")
                class_names.add(class_name)
            class_names.add(class_name)

            # Check if it's an accident using ACCIDENT_CLASSES (car and person never trigger alerts)
            now = time.time()
            if class_name.lower() in ACCIDENT_CLASSES and (now - last_accident_time >= config.cooldown_seconds):
                last_accident_time = now
                image_filename = f"image_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                image_path = os.path.join(config.report_images_dir, image_filename)
                cv2.imwrite(image_path, frame)  # Save the image
                list_of_images.append(image_path)
                accident_detectd = True

    if accident_detectd:
        report_filename = generate_report(' '.join(class_names), list_of_images)
        st.toast(f"Accident Detected: {' '.join(class_names)}. Report saved: {report_filename}", icon="🚨")
                
    cap.release()
    st.write("Video playback complete.")

# Function to show reports
def show_reports():
    st.title("Accident Reports")

    if st.session_state.report_filenames:
        for report_filename, image_paths in st.session_state.report_filenames:
            report_path = os.path.join(config.reports_dir, report_filename)
            report_df = pd.read_csv(report_path)

            # Display the report title
            st.subheader(f"{report_df['Class'][0]} in {str(report_df['Time'][0])}")

            # Display an expander to toggle the description visibility
            with st.expander("Report details"):
                num_images_to_show = min(2, len(image_paths))
                selected_images = image_paths[0], image_paths[-1]

                # Display the selected images side by side using columns
                cols = st.columns(num_images_to_show)
                for i in range(num_images_to_show):
                    cols[i].image(selected_images[i], use_column_width=True)

                # Create a unique key for each report's description button
                description_key = f"description_{report_filename}"
                description_exists = 'Description' in report_df.columns and pd.notnull(report_df['Description'][0])
                if description_exists:
                    st.write("**Description:**", report_df['Description'][0])
                else:
                    # Initialize the session state for description if it doesn't exist
                    if description_key not in st.session_state:
                        st.session_state[description_key] = ""

                    # Button for generating description
                    if st.button(f"Generate Description for {report_df['Class'][0]}_{str(report_df['Time'][0])}"):
                        # Generate a description for the report
                        description = generate_accident_report(
                            selected_images, f"{report_df['Class'][0]}", report_df["Time"][0]
                        )
                        st.session_state[description_key] = description
                        
                        # Update the report DataFrame with the new description
                        report_df['Description'] = description
                        report_df.to_csv(report_path, index=False)
                        
                        st.success("Description generated successfully!")

                    # Display the description if it has been generated
                    if st.session_state[description_key]:
                        st.write("**Description:**", st.session_state[description_key])
    else:
        st.write("No accident reports yet.")

# Streamlit app UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Video & Prediction", "Reports"])

if page == "Video & Prediction":
    st.title("CCTV Accident Detection")
    with st.expander('Live Cam Detection'):
        st.write('This is live cam example')
    # File uploader for video input
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "gif"])

    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        video_path = os.path.join(config.temp_dir, 'uploaded_video.mp4')
        with open(video_path, 'wb') as f:
            f.write(uploaded_video.read())

        # Display the annotated video using the placeholder 
        display_annotated_video(video_path)

elif page == "Reports":
    show_reports()
