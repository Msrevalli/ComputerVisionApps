import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano for fast inference

# Streamlit UI
st.title("🔴 Real-Time Object Detection using YOLOv8 🚀")
st.write("Click 'Start' to begin object detection and 'Stop' to end the video stream.")

# Buttons for controlling the stream
start_button = st.button("Start Video Stream")
stop_button = st.button("Stop Video Stream")

# Initialize Video Capture
cap = None

# Check if the Start button is pressed
if start_button:
    cap = cv2.VideoCapture(0)  # Start webcam video capture
    frame_window = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture video")
            break

        # Run YOLOv8 object detection
        results = model(frame)

        # Process detection results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = float(box.conf[0])  # Confidence score
                class_id = int(box.cls[0])  # Class ID
                label = f"{model.names[class_id]}: {conf:.2f}"  # Class label

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to RGB (for Streamlit)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

        # Stop button logic
        if stop_button:
            cap.release()
            cv2.destroyAllWindows()
            st.warning("Video Stream Stopped.")
            break

# Release resources if Stop button is pressed
if stop_button and cap is not None:
    cap.release()
    cv2.destroyAllWindows()
