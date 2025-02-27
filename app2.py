import streamlit as st
import cv2
import numpy as np
import asyncio
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Fix asyncio event loop issue for Streamlit Cloud
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

st.title("🎥 Real-Time Object Detection with YOLOv8")
st.write("This app uses WebRTC to access your webcam in Streamlit Cloud.")

# Custom Video Processor for Object Detection
class ObjectDetection(VideoProcessorBase):
    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")  # Convert to OpenCV format
        results = model(frame)  # Run YOLO detection

        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0]  # Get bounding box coordinates
            conf = result.conf[0]  # Confidence score
            cls = int(result.cls[0])  # Class index
            label = model.names[cls]  # Get class name

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame

# Start WebRTC Video Streaming with STUN/TURN servers (for Streamlit Cloud)
webrtc_streamer(
    key="object-detection",
    video_processor_factory=ObjectDetection,
    rtc_configuration={
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},  # Google's Public STUN Server
            {"urls": "turn:openrelay.metered.ca:80", "username": "openrelay", "credential": "openrelay"}  # Free TURN Server
        ]
    }
)
