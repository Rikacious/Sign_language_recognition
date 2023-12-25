import os
import cv2
import json
import numpy as np
from threading import Thread

from tensorflow.keras import models

from video import Video


class handNumber():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        jsonFile = open('settings.json')
        settings = json.load(jsonFile)
        jsonFile.close()
        
        self.tipIds = [4, 8, 12, 16, 20]
        self.numVid = Video(hands=1, detectionCon=0.6, seqLength=1)
        self.actions = np.array(settings['actions']['number'])
        self.model = models.load_model(os.path.join(
            settings['modelsDir'], 
            settings['models']['number']
        ))

    def startNumPrediction(self, image, errFunc=None, opFunc=None, showFPS=False):
        if not self.numVid.checkVisibility(image): # Checking Frame Visibility
            errText = "Video too Dark."
            if errFunc == None:
                image = self.numVid.showText(image, errText, color=(0,0,255))
            else:
                errFunc(errText)
        else:
            self.numVid.getHandPosition(image) # Track Hand Position with MediaPipe

            if(self.getHandNumPoints()):
                # image = tracker.handsFinder(image) # Showing Hand Links in the Frame
                image = self.numVid.showBBox(image) # Showing Hand Bounding Box in the Frame
                Thread(target=lambda: self.getNumPrediction(opFunc)).start() # Starting Prediction in Annother Thread
                if opFunc == None:
                    image = self.numVid.showText(image, predict=True)
            else:
                errText = "Hand not Detected Properly."
                if errFunc == None:
                    image = self.numVid.showText(image, errText, color=(0, 0, 255))
                else:
                    errFunc(errText)

        if showFPS:
            image = self.numVid.showFPS(image) # Adding FPS to the Image

        return image

    def getHandNumPoints(self):
        visibility = False
        handPoints = []

        if self.numVid.results and self.numVid.results.multi_hand_landmarks:
            for idx, hand_landmark in enumerate(self.numVid.results.multi_hand_landmarks):
                points = self.numVid.getHandPoints(hand_landmark)
                handPoints.append(points)
                visibility = (np.array(points).flatten()).any()

            self.numVid.keyPoints.append(np.array(handPoints))
            self.numVid.keyPoints = self.numVid.keyPoints[(self.numVid.seqLength * -1):]
        
        return(visibility)

    def getFingureIsUp(self, points):
        fingers = [0, 0, 0, 0, 0]
        points = [(point[0] * 640, point[1] * 480) for point in points]

        # All Four Fingures exept Thumb
        for id in range(1, 5):
            fingers[id] =  (points[self.tipIds[id]][1] < points[self.tipIds[id] - 1][1]) and 1 or 0

        # For Thumb Only
        label = self.numVid.results.multi_handedness[0].classification[0].label
        if(label == "Right"):
            fingers[0] = (points[self.tipIds[0]][0] < points[self.tipIds[0] - 1][0]) and 1 or 0
        else:
            fingers[0] = (points[self.tipIds[0]][0] > points[self.tipIds[0] - 1][0]) and 1 or 0

        return fingers

    def getNumPrediction(self, callBack=None):
        if len(self.numVid.keyPoints) == self.numVid.seqLength:
            fUpList = self.getFingureIsUp(self.numVid.keyPoints[0][0])
            predictStr = ""

            if (fUpList == [0, 1, 0, 0, 0]):
                predictStr = "ONE"
            elif (fUpList == [0, 1, 1, 0, 0]):
                predictStr = "TWO"
            elif (fUpList == [0, 0, 1, 1, 1]):
                predictStr = "THREE"
            elif (fUpList == [0, 1, 1, 1, 1]):
                predictStr = "FOUR"
            elif (fUpList == [1, 1, 1, 1, 1]):
                predictStr = "FIVE"
            elif (fUpList == [0, 1, 1, 1, 0]):
                predictStr = "SIX"
            elif (fUpList == [0, 1, 1, 0, 1]):
                predictStr = "SEVEN"
            elif (fUpList == [0, 1, 0, 1, 1]):
                predictStr = "EIGHT"
            elif (fUpList == [0, 0, 1, 1, 1]):
                predictStr = "NINE"
            elif (fUpList == [0, 0, 0, 0, 0]):
                predictStr = "ZERO"

            if predictStr != self.numVid.lastPredict:
                # print(predictStr)
                self.numVid.lastPredict = predictStr
                if callBack != None:
                    callBack(predictStr)

            '''
            predictPoints = self.numVid.keyPoints[0].flatten()
            predict = self.model.predict(np.expand_dims(predictPoints, axis=0))[0]
            maxPredict = np.argmax(predict)
            # print(predict)
            predictStr = self.actions[maxPredict]

            if predict[maxPredict] > 0.8 and predictStr != self.numVid.lastPredict:
                    self.numVid.lastPredict = predictStr
                    if callBack != None:
                        callBack(predictStr)
            '''
            


def main():
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vc.set(cv2.CAP_PROP_FPS, 30)
    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    tracker = handNumber()

    while vc.isOpened():
        success, frame = vc.read()
        image = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect
        
        image = tracker.startNumPrediction(image, showFPS=True)

        cv2.imshow("Hand Gesture Detection", image)
        
        if cv2.waitKey(1) == 27: # exit on ESC
            break
    
    vc.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()