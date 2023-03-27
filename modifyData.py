import os
import cv2
import json
import time
import numpy as np
import mediapipe as mp
from threading import Thread





actions = [
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "ZERO"
]

for action in actions:
    os.makedirs(os.path.join('MP_DATA/NUMBER', action))
    
    for sequence in range(30):
        window = np.load(os.path.join('MP_DATA/CHAR', action, f"{sequence}.npy"))
        print(np.array(window).shape, np.array(window[2:7]).shape)

        nPyPath = os.path.join('MP_DATA/NUMBER', action, str(sequence))
        np.save(nPyPath, np.array(window[2:7]))