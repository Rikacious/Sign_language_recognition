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

        self.numVid = Video(hands=1, detectionCon=0.6, seqLength=5)
        self.actions = np.array(settings['actions']['number'])
        self.model = models.load_model(os.path.join(
            settings['modelsDir'], 
            settings['models']['number']
        ))

    def startNumPrediction(self, image, errFunc=None, opFunc=None, showFPS=False):
        if not self.numVid.checkVisibility(image): # Checking Frame Visibility
            if errFunc == None:
                image = self.numVid.showText(image, isDark=True)
            else:
                errFunc("Video too Dark.")
        else:
            self.numVid.getHandPosition(image) # Track Hand Position with MediaPipe

            if(self.getHandNumPoints()):
                # image = tracker.handsFinder(image) # Showing Hand Links in the Frame
                Thread(target=lambda: self.getNumPrediction(opFunc)).start() # Starting Prediction in Annother Thread
                if opFunc == None:
                    image = self.numVid.showText(image, output=True)
            else:
                if errFunc == None:
                    image = self.numVid.showText(image, isDark=False, isHandVisible=False)
                else:
                    errFunc("Hand Not Visible.")

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
                visibility = points.any()

            self.numVid.keyPoints.append(np.array(handPoints).flatten())
            self.numVid.keyPoints = self.numVid.keyPoints[(self.numVid.seqLength * -1):]
        
        return(visibility)

    def getNumPrediction(self, callBack=None):
        if len(self.numVid.keyPoints) == self.numVid.seqLength:
            predict = self.model.predict(np.expand_dims(self.numVid.keyPoints, axis=0))[0]
            maxPredict = np.argmax(predict)
            # print(predict)
            predictStr = self.actions[maxPredict]

            if predict[maxPredict] > 0.8 and predictStr != self.numVid.lastPredict:
                    self.numVid.lastPredict = predictStr
                    if callBack != None:
                        callBack(predictStr)


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