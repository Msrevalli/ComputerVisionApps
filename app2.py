import streamlit as st
import cv2
import numpy as np
import asyncio
from ultralytics import YOLO

# Fix asyncio event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load YOLO model
model = YOLO("yolov8n.pt")

st.title("🎥 Real-Time Object Detection with YOLOv8")
st.write("Start webcam to detect objects in real-time.")

# Function to find a working camera
def get_camera():
    for cam_id in range(3):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap, cam_id
    return None, -1

# Start & Stop buttons
start_button = st.button("▶ Start")
stop_button = st.button("⏹ Stop")

# Start webcam if button is clicked
if start_button:
    cap, selected_cam = get_camera()

    if cap is None:
        st.error("❌ No available cameras found. Try enabling permissions or restarting your device.")
    else:
        st.success(f"📷 Using Camera Index {selected_cam}")
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                st.warning("⏹ Webcam Stopped!")
                break

            frame = cv2.resize(frame, (640, 480))  # Resize for faster processing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame)  # Run YOLO detection

            for result in results[0].boxes:
                x1, y1, x2, y2 = result.xyxy[0]
                conf = result.conf[0]
                cls = int(result.cls[0])
                label = model.names[cls]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            stframe.image(frame, channels="RGB", use_container_width=True)

        cap.release()
