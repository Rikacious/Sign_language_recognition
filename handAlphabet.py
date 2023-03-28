import os
import cv2
import json
import numpy as np
from threading import Thread

from tensorflow.keras import models

from video import Video


class handAlphabet():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        jsonFile = open('settings.json')
        settings = json.load(jsonFile)
        jsonFile.close()

        self.alpVid = Video(hands=1, detectionCon=0.6, seqLength=5)
        self.actions = np.array(settings['actions']['alphabet'])
        self.model = models.load_model(os.path.join(
            settings['modelsDir'], 
            settings['models']['alphabet']
        ))

    def startAlpPrediction(self, image, errFunc=None, opFunc=None, showFPS=False):
        if not self.alpVid.checkVisibility(image): # Checking Frame Visibility
            if errFunc == None:
                image = self.alpVid.showText(image, isDark=True)
            else:
                errFunc("Video too Dark.")
        else:
            self.alpVid.getHandPosition(image) # Track Hand Position with MediaPipe

            if(self.getHandAlpPoints()):
                # image = tracker.handsFinder(image) # Showing Hand Links in the Frame
                Thread(target=lambda: self.getAlpPrediction(opFunc)).start() # Starting Prediction in Annother Thread
                if opFunc == None:
                    image = self.alpVid.showText(image, output=True)
            else:
                if errFunc == None:
                    image = self.alpVid.showText(image, isDark=False, isHandVisible=False)
                else:
                    errFunc("Hand Not Visible.")

        if showFPS:
            image = self.alpVid.showFPS(image) # Adding FPS to the Image

        return image

    def getHandAlpPoints(self):
        visibility = False
        handPoints = []

        if self.alpVid.results and self.alpVid.results.multi_hand_landmarks:
            for idx, hand_landmark in enumerate(self.alpVid.results.multi_hand_landmarks):
                points = self.alpVid.getHandPoints(hand_landmark)
                handPoints.append(points.flatten())
                visibility = points.any()

            self.alpVid.keyPoints.append(np.array(handPoints).flatten())
            self.alpVid.keyPoints = self.alpVid.keyPoints[(self.alpVid.seqLength * -1):]
        
        return(visibility)

    def getAlpPrediction(self, callBack=None):
        if len(self.alpVid.keyPoints) == self.alpVid.seqLength:
            predict = self.model.predict(np.expand_dims(self.alpVid.keyPoints, axis=0))[0]
            maxPredict = np.argmax(predict)
            # print(predict)
            predictStr = self.actions[maxPredict]

            if predict[maxPredict] > 0.8 and predictStr != self.alpVid.lastPredict:
                    self.alpVid.lastPredict = predictStr
                    if callBack != None:
                        callBack(predictStr)


def main():
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vc.set(cv2.CAP_PROP_FPS, 30)
    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    tracker = handAlphabet()

    while vc.isOpened():
        success, frame = vc.read()
        image = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect
        
        image = tracker.startAlpPrediction(image, showFPS=True)

        cv2.imshow("Hand Gesture Detection", image)
        
        if cv2.waitKey(1) == 27: # exit on ESC
            break
    
    vc.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()