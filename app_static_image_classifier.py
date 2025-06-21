import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("model.h5")

st.title("Sign Language Recognition")

uploaded_file = st.file_uploader("Upload an image of a hand sign", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize image to model's expected input size
    img = img.resize((64, 64))  # Change this size if needed
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)  # Adjust based on model input

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.subheader(f"Predicted Sign Class: {predicted_class}")
