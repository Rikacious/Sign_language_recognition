import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model1.h5")

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Prediction function
def predict_sequence(seq):
    input_data = np.expand_dims(seq, axis=0)  # Shape: (1, 20, 126)
    prediction = model.predict(input_data)[0]
    gesture = np.argmax(prediction)
    confidence = np.max(prediction)
    return gesture, confidence

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.sequence = []

    def extract_keypoints(self, hand_landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        h, w, _ = image.shape
        cv2.rectangle(image, (w//2 - 100, h//2 - 100), (w//2 + 100, h//2 + 100), (0, 255, 255), 2)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                keypoints = self.extract_keypoints(handLms)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-20:]
                mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

                if len(self.sequence) == 20:
                    gesture, confidence = predict_sequence(self.sequence)
                    cv2.putText(image, f"Gesture: {gesture} ({confidence:.2f})", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image

st.title("Sign Language Recognition - Streamlit App")

webrtc_streamer(key="sign-lang", video_processor_factory=VideoProcessor)
