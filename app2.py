import streamlit as st
import cv2
import numpy as np
import asyncio
from ultralytics import YOLO
import torch

# Fix asyncio event loop issue for Streamlit Cloud
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Fix PyTorch class registration issue
torch._classes

# Load YOLOv8 Model (Nano for better speed)
model = YOLO("yolov8n.pt")

st.title("🎥 Real-Time Object Detection with YOLOv8")
st.write("Select a camera and click 'Start' for real-time object detection.")

# Sidebar camera selection
camera_option = st.sidebar.radio("Choose Camera:", ("Front Camera", "Back Camera"))
camera_index = 0 if camera_option == "Front Camera" else 1

# Start & Stop buttons
start_button = st.button("▶ Start")
stop_button = st.button("⏹ Stop")

# Function for object detection
def detect_objects(frame):
    results = model(frame)  # Run YOLO detection

    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]  # Get bounding box coordinates
        conf = result.conf[0]  # Confidence score
        cls = int(result.cls[0])  # Class index
        label = model.names[cls]  # Get class name
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

# Start Webcam
if start_button:
    cap = None
    for cam_id in range(3):  # Try multiple camera indexes (0, 1, 2)
        cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)  # Use V4L2 backend for Linux
        if cap.isOpened():
            break

    stframe = st.empty()

    if cap is None or not cap.isOpened():
        st.error("❌ Failed to access webcam. Try another camera index.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret or stop_button:
                st.warning("⏹ Webcam Stopped!")
                break

            frame = detect_objects(cv2.resize(frame, (640, 480)))  # Resize for faster processing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert for Streamlit display
            stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()
