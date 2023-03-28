import cv2
import time
import numpy as np
import mediapipe as mp


class Video:
    def __init__(self, hands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5, seqLength=10, **kwargs):
        super().__init__(**kwargs)

        self.currentTime = 0
        self.previousTime = 0
        self.keyPoints = []
        self.lastPredict = None
        self.seqLength = seqLength
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(False, hands, modelComplexity, detectionCon, trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = False

    def showText(self, image, output=False, isDark=True, isHandVisible=False):
        fontScale = 1
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        gap = 0.1
        boxStart = (int(image.shape[1] * gap), int(image.shape[0] * gap))

        if output:
            color = (245, 117, 16)
            text = self.lastPredict != None and self.lastPredict or f"Detection Started {self.seqLength - len(self.keyPoints)}"
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

    def checkVisibility(self, image):
        blur = cv2.blur(image, (5,5))
        mean = np.mean(blur)
        return(mean > 80 and True or False)

    def getHandPosition(self, image):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

    def getHandPoints(self, hand_landmark):
        threshold = {"X": 0.0075, "Y": 0.0175}

        points = np.array([
            [res.x, res.y, res.z] for res in hand_landmark.landmark
        ])

        x0, y0, _ = points[0]
        x5, y5, _ = points[5]
        x17, y17, _ = points[17]

        disX = (x5 - x17)**2 + (y5 - y17)**2  #0.0081
        disY = ((x5 + x17)/2 - x0)**2 + ((y5 + y17)/2 - y0)**2  #0.024
        # print(disX, disY)

        if (disX >= threshold["X"] or disY >= threshold["Y"]):
            return np.array(points)
        else:
            return np.zeros(21*3)

    def handsFinder(self, image):
        if self.results.multi_hand_landmarks:
            for handLimbs in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(image, handLimbs, self.mpHands.HAND_CONNECTIONS)

        return image

