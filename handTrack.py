import os
import cv2
import time
import math
import json
import numpy as np
import mediapipe as mp
from threading import Thread

from tensorflow.keras import models


class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        jsonFile = open('settings.json')
        settings = json.load(jsonFile)
        jsonFile.close()

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.currentTime = 0
        self.previousTime = 0
        self.sentence = []
        self.keyPoints = []
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.sequenceLength = settings['sequenceLength']
        self.actions = np.array(settings['actions'])
        self.model = models.load_model(os.path.join(settings['modelsDir'], settings['lastModel']))

    def showTextOnScreen(self, image, output=False, isDark=True, isHandVisible=False):
        fontScale = 1
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        gap = 0.1
        boxStart = (int(image.shape[1] * gap), int(image.shape[0] * gap))

        if output:
            color = (245, 117, 16)
            text = len(self.sentence) > 0 and self.sentence[-1] or f"Detection Started {10 - len(self.keyPoints)}"
        else:
            if isDark:
                color = (0, 0, 255)
                text = "Video is Too Dark"
            elif not isHandVisible:
                color = (0, 0, 255)
                text = "Hand Not Visible"

        textSize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = int((image.shape[1] - textSize[0]) / 2)
        textY = int((boxStart[1] + textSize[1]) / 2)

        cv2.putText(image, text, (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA)

        return image

    def showFPS(self, image):
        self.currentTime = time.time()
        fps = 1 / (self.currentTime - self.previousTime)
        self.previousTime = self.currentTime
        # Displaying FPS on the image
        cv2.putText(image, str(int(fps))+" FPS", (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,0,255), 1)

        return image

    def checkFrameVisibility(self, image):
        blur = cv2.blur(image, (5,5))
        mean = np.mean(blur)
        return(mean > 80 and True or False)

    def getHandPosition(self, image):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

    def getHandVisibility(self):
        visibility = 0

        if self.results.multi_hand_landmarks:
            handMarks = self.results.multi_hand_landmarks[0]
            x5, y5 = tuple(np.multiply(np.array((
                handMarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_MCP].x,
                handMarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_MCP].y
            )), [640, 480]).astype(int))
            x17, y17 = tuple(np.multiply(np.array((
                handMarks.landmark[self.mpHands.HandLandmark.PINKY_MCP].x,
                handMarks.landmark[self.mpHands.HandLandmark.PINKY_MCP].y
            )), [640, 480]).astype(int))

            visibility = math.sqrt((x17-x5)**2 + (y17-y5)**2)
            # print(visibility)
        
        return(visibility > 50 and True or False)

    def handsFinder(self, image):
        if self.results.multi_hand_landmarks:
            handLimbs = self.results.multi_hand_landmarks[0]
            self.mpDraw.draw_landmarks(image, handLimbs, self.mpHands.HAND_CONNECTIONS)

        return image

    def getKeyPoints(self):
        self.keyPoints.append(np.array([
            [res.x, res.y, res.z] for res in self.results.multi_hand_landmarks[0].landmark
        ]).flatten())
        self.keyPoints = self.keyPoints[(self.sequenceLength * -1):]

    def getPrediction(self):
        if len(self.keyPoints) == self.sequenceLength:
            predict = self.model.predict(np.expand_dims(self.keyPoints, axis=0))[0]
            maxPredict = np.argmax(predict)
            # print(predict)

            if(predict[maxPredict] > 0.8):
                self.sentence.append(self.actions[maxPredict])


def main():
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vc.set(cv2.CAP_PROP_FPS, 30)
    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    tracker = handTracker(maxHands=2, detectionCon=0.6)

    while vc.isOpened():
        success, frame = vc.read()
        frame = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect
        frameVisible = tracker.checkFrameVisibility(frame) # Checking Frame Visibility

        if not frameVisible:
            image = tracker.showTextOnScreen(frame, isDark=True)
        else:
            tracker.getHandPosition(frame) # Track Hand Position with MediaPipe
            handVisible = tracker.getHandVisibility() # Checking Hand Visibility

            if(handVisible):
                image = tracker.handsFinder(frame) # Showing Hand Links in the Frame

                tracker.getKeyPoints()
                Thread(target=tracker.getPrediction).start()
                # tracker.getPrediction()
                image = tracker.showTextOnScreen(image, output=True)
            else:
                image = tracker.showTextOnScreen(frame, isDark=False, isHandVisible=False)

        image = tracker.showFPS(image) # Adding FPS to the Image
        cv2.imshow("Hand Gesture Detection", image)
        
        if cv2.waitKey(1) == 27: # exit on ESC
            break
    
    vc.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()