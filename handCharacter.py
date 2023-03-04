import os
import cv2
import json
import numpy as np
from threading import Thread

from tensorflow.keras import models

from video import Video


class handCharacter():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        jsonFile = open('settings.json')
        settings = json.load(jsonFile)
        jsonFile.close()

        self.charVid = Video(hands=2, detectionCon=0.6, seqLength=15) # seqLength Updated to 20
        self.actions = np.array(settings['actions']['char'])
        self.model = models.load_model(os.path.join(
            settings['modelsDir'], 
            settings['models']['char']
        ))

    def startCharPrediction(self, image, errFunc=None, opFunc=None, showFPS=False):
        if not self.charVid.checkVisibility(image): # Checking Frame Visibility
            if errFunc == None:
                image = self.charVid.showText(image, isDark=True)
            else:
                errFunc("Video too Dark.")
        else:
            self.charVid.getHandPosition(image) # Track Hand Position with MediaPipe

            if(self.getHandCharPoints()):
                # image = tracker.handsFinder(image) # Showing Hand Links in the Frame
                Thread(target=lambda: self.getCharPrediction(opFunc)).start() # Starting Prediction in Annother Thread
                if opFunc == None:
                    image = self.charVid.showText(image, output=True)
            else:
                if errFunc == None:
                    image = self.charVid.showText(image, isDark=False, isHandVisible=False)
                else:
                    errFunc("Hand Not Visible.")

        if showFPS:
            image = self.charVid.showFPS(image) # Adding FPS to the Image

        return image

    def getHandCharPoints(self):
        visibility = False
        handPoints = []

        if self.charVid.results and self.charVid.results.multi_hand_landmarks:
            for idx, hand_landmark in enumerate(self.charVid.results.multi_hand_landmarks):
                points = self.charVid.getHandPoints(hand_landmark)
                handPoints.append(points)
                visibility = points.any()

            self.charVid.keyPoints.append(handPoints)
            self.charVid.keyPoints = self.charVid.keyPoints[(self.charVid.seqLength * -1):]
        
        return(visibility)

    def getCharPrediction(self, callBack=None):
        if len(self.charVid.keyPoints) == self.charVid.seqLength:
            predict = self.model.predict(np.expand_dims(self.charVid.keyPoints, axis=0))[0]
            maxPredict = np.argmax(predict)
            # print(predict)
            predictStr = self.actions[maxPredict]

            if predict[maxPredict] > 0.8 and predictStr != self.charVid.lastPredict:
                    self.charVid.lastPredict = predictStr
                    if callBack != None:
                        callBack(predictStr)


def main():
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vc.set(cv2.CAP_PROP_FPS, 30)
    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    tracker = handCharacter()

    while vc.isOpened():
        success, frame = vc.read()
        image = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect
        
        image = tracker.startCharPrediction(image)

        cv2.imshow("Hand Gesture Detection", image)
        
        if cv2.waitKey(1) == 27: # exit on ESC
            break
    
    vc.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()