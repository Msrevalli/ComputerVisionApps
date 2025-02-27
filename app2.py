import streamlit as st
import cv2
import numpy as np
import asyncio
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("🎥 Real-Time Object Detection with YOLOv8")
st.write("Webcam stream will appear below. Streamlit WebRTC allows camera access in Streamlit Cloud.")

# Fix asyncio event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Custom video processor for object detection
class ObjectDetection(VideoProcessorBase):
    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")  # Convert to OpenCV format
        results = model(frame)  # Run YOLO detection

        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0]
            conf = result.conf[0]
            cls = int(result.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame

# Start WebRTC video streaming (Fixed version)
webrtc_streamer(key="object-detection", video_processor_factory=ObjectDetection)
