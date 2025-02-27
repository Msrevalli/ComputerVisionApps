import streamlit as st
import os
os.environ["LD_LIBRARY_PATH"] = "/usr/lib64:/usr/lib"

os.environ["YOLO_VERBOSE"] = "False"
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")  # Using YOLOv8 nano for real-time detection

st.title("🎥 Live Object Detection with YOLOv8")
st.write("Select a camera and click 'Start' for real-time object detection.")

# Camera selection (0 = Front Camera, 1 = Back Camera)
camera_option = st.radio("Choose Camera:", ("Front Camera", "Back Camera"))
camera_index = 0 if camera_option == "Front Camera" else 1

# Start/Stop button
start_button = st.button("▶ Start")
stop_button = st.button("⏹ Stop")

# Function to process frames
def detect_objects(frame):
    results = model(frame)  # Run YOLO detection

    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]  # Bounding box coordinates
        conf = result.conf[0]  # Confidence score
        cls = int(result.cls[0])  # Class index
        label = model.names[cls]  # Get class name

        # Draw bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

# Webcam access and object detection
if start_button:
    cap = cv2.VideoCapture(camera_index)  # Open the selected camera
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_button:  # Check if stop button is pressed
            st.warning("⏹ Stopped Webcam!")
            break

        frame = detect_objects(frame)  # Process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert for Streamlit display

        stframe.image(frame, channels="RGB", use_container_width=True)

        time.sleep(0.03)  # Small delay for smooth streaming

    cap.release()
