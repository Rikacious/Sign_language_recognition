import numpy as np
import cv2
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

class Predictor:
    def __init__(self):
        model_path = os.path.join("Trained_Models", "model1.h5")  # ✅ Correct path
        self.model = load_model(model_path)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.sequence = []

    def extract_keypoints(self, hand_landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

    def __call__(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                keypoints = self.extract_keypoints(handLms)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-20:]
                self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)

            if len(self.sequence) == 20:
                input_data = np.expand_dims(self.sequence, axis=0)
                prediction = self.model.predict(input_data)[0]
                gesture = np.argmax(prediction)
                confidence = np.max(prediction)
                cv2.putText(frame, f"Prediction: {gesture} ({confidence:.2f})", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        h, w, _ = frame.shape
        cv2.rectangle(frame, (w//2 - 100, h//2 - 100), (w//2 + 100, h//2 + 100), (0, 255, 255), 3)
        cv2.putText(frame, "ROI", (w//2 - 30, h//2 - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame
