import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os


class dataCollection:
    def __init__(self, detectionCon=0.8, modelComplexity=1, trackCon=0.5):
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.DATA_PATH = os.path.join('MP_DATA')
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(False, 2,self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def createDirectories(self, actions, noSequence):
        for action in actions:
            for sequence in range(noSequence):
                try:
                    os.makedirs(os.path.join(self.DATA_PATH, action, str(sequence)))
                except:
                    pass

    def showTextOnScreen(self, image, isDark=True, isHandVisible=False):
        fontScale = 1
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        gap = 0.1
        boxStart = (int(image.shape[1] * gap), int(image.shape[0] * gap))

        if isDark:
            color = (0, 0, 255)
            text = "Video is Too Dark"
        else:
            if isHandVisible:
                color = (0, 255, 0)
                text = "Hand is Visible Properly"
            else:
                color = (0, 0, 255)
                text = "Hand Not Visible"

        textSize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = int((image.shape[1] - textSize[0]) / 2)
        textY = int((boxStart[1] + textSize[1]) / 2)

        cv2.putText(image, text, (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA)

        return image

    def getHandPosition(self, image):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

    def handsFinder(self, image):
        if self.results.multi_hand_landmarks:
            handLimbs = self.results.multi_hand_landmarks[0]
            self.mpDraw.draw_landmarks(image, handLimbs, self.mpHands.HAND_CONNECTIONS)

        return image

    def storeKeyPoints(self, action, sequence, frameNum):
        keyPoints = np.array([
            [res.x, res.y, res.z] for res in self.results.multi_hand_landmarks[0].landmark
        ]).flatten() if self.results.multi_hand_landmarks else np.zeros(21*3)

        npyPath = os.path.join(self.DATA_PATH, action, str(sequence), str(frameNum))
        np.save(npyPath, keyPoints)



def main():
    noSequence = 30
    sequenceLength = 10
    actions = ["ONE", "TWO", "THREE"]

    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        collect = dataCollection()
        # collect.createDirectories(actions, noSequence)
        
        for action in actions:
            for sequence in range(noSequence):
                for frameNum in range(sequenceLength):

                    success, frame = vc.read()
                    frame = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect

                    collect.getHandPosition(frame) # Track Hand Position with MediaPipe
                    image = collect.handsFinder(frame) # Showing Hand Links in the Frame

                    if frameNum == 0:
                        print("First Frame", action, sequence)
                        cv2.imshow("Collecting Hand Data", image)
                        cv2.waitKey(3000)
                    else:
                        print("Other Frames", action,sequence)
                        cv2.imshow("Collecting Hand Data", image)

                    collect.storeKeyPoints(action, sequence, frameNum)
        
                    if cv2.waitKey(10) == 27: # exit on ESC
                        break
        
        vc.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()