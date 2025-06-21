import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from pathlib import Path

# Path to directory containing models
model_dir = Path("Trained_Models")

# Test image (replace with your own test image path)
test_img_path = "test_image.jpg"  # Make sure this exists

# Image size expected by the model
target_size = (64, 64)  # Adjust if your model uses a different size

# Load and preprocess the image
img = image.load_img(test_img_path, target_size=target_size)
img_array = image.img_to_array(img) / 255.0
img_array = img_array.reshape(1, *target_size, 3)

# Optional: Class labels (replace with actual if available)
class_labels = [chr(i) for i in range(65, 91)]  # 'A' to 'Z'

# Iterate over all .h5 models in the folder
for model_file in model_dir.glob("*.h5"):
    try:
        print(f"\n🔍 Evaluating model: {model_file.name}")
        model = tf.keras.models.load_model(model_file)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)

        label = class_labels[predicted_index] if predicted_index < len(class_labels) else str(predicted_index)
        print(f"🧠 Prediction: {label} (class {predicted_index}), Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"❌ Failed to evaluate {model_file.name}: {e}")
