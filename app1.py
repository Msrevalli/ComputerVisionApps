import streamlit as st
import os
os.environ["LD_LIBRARY_PATH"] = "/usr/lib64:/usr/lib"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to apply image processing techniques
def process_image(image, filter_type):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    if filter_type == "Original":
        return gray_image
    elif filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(gray_image, (5, 5), 0)
    elif filter_type == "Median Blur":
        return cv2.medianBlur(gray_image, 5)
    elif filter_type == "Histogram Equalization":
        return cv2.equalizeHist(gray_image)
    elif filter_type == "Sobel Edge Detection":
        return cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    elif filter_type == "Canny Edge Detection":
        return cv2.Canny(gray_image, 100, 200)
    return gray_image

# Streamlit UI
st.title("🖼️ Image Filtering and Enhancement App")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Select Filter
    filter_option = st.sidebar.radio(
        "Select Image Processing Technique:",
        ("Original", "Gaussian Blur", "Median Blur", "Histogram Equalization", "Sobel Edge Detection", "Canny Edge Detection")
    )

    # Process Image
    processed_image = process_image(image, filter_option)

    # Display images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(processed_image, caption=f"{filter_option} Applied", use_container_width=True, clamp=True)

    st.write(f"**Applied Filter:** {filter_option}")

# Footer
st.markdown("Developed using Streamlit & OpenCV")

