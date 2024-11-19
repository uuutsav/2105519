import streamlit as st
from PIL import Image
import numpy as np
#import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
#import cv2

# Load the pre-trained model
model = load_model('autoencode.h5')

# Constants
IMG_SIZE = 160  # The input size expected by the model

# Streamlit interface
st.title("Grayscale to Color Image Colorization")

# Upload image
uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the original image
    original_image = Image.open(uploaded_file)

    # Convert image to RGB if not already in RGB
    image_rgb = original_image.convert("RGB")
    
    # Resize the image to the appropriate input size for the model
    image_resized = image_rgb.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image_resized).astype("float32") / 255.0  # Normalize to [0, 1]
    
    # Reshape to add batch dimension
    input_image = np.reshape(image_array, (1, IMG_SIZE, IMG_SIZE, 3))

    # Predict the colorized image
    predicted_image = model.predict(input_image)
    predicted_image = np.clip(predicted_image, 0.0, 1.0).reshape(IMG_SIZE, IMG_SIZE, 3)

    # Display images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(predicted_image, caption="Colorized Image", use_column_width=True)
