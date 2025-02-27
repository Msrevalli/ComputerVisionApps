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
