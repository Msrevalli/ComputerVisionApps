import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # You can use "yolov8s.pt" for better accuracy

# Streamlit UI
st.title("📹 Real-Time Object Detection using YOLOv8 🚀")
st.write("Enable your webcam and detect objects in real-time.")

# WebRTC Configuration (Fix Streamlit Cloud Issues)
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Video Transformer Class
class YOLOv8VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 object detection
        results = self.model(img)

        # Process detection results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = float(box.conf[0])  # Confidence score
                class_id = int(box.cls[0])  # Class ID
                label = f"{model.names[class_id]}: {conf:.2f}"  # Class label

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

# WebRTC Streamer with RTC Configuration
webrtc_streamer(
    key="yolo-webcam",
    video_transformer_factory=YOLOv8VideoTransformer,
    rtc_configuration=rtc_config  # Enables WebRTC in Streamlit Cloud
)
