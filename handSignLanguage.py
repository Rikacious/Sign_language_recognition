import os
import cv2
import json
import numpy as np
from threading import Thread

from tensorflow.keras import models

from video import Video


class handSign():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        jsonFile = open('settings.json')
        settings = json.load(jsonFile)
        jsonFile.close()

        self.signVid = Video(hands=2, detectionCon=0.6, seqLength=15) # seqLength Updated to 20
        self.actions = np.array(settings['actions']['sign'])
        self.model = models.load_model(os.path.join(
            settings['modelsDir'], 
            settings['models']['sign']
        ))

    def startSignPrediction(self, image, errFunc=None, opFunc=None, showFPS=False):
        if not self.signVid.checkVisibility(image): # Checking Frame Visibility
            if errFunc == None:
                image = self.signVid.showText(image, isDark=True)
            else:
                errFunc("Video too Dark.")
        else:
            self.signVid.getHandPosition(image) # Track Hand Position with MediaPipe

            if(self.getHandSignPoints()):
                # image = tracker.handsFinder(image) # Showing Hand Links in the Frame
                Thread(target=lambda: self.getSignPrediction(opFunc)).start() # Starting Prediction in Annother Thread
                if opFunc == None:
                    image = self.signVid.showText(image, output=True)
            else:
                if errFunc == None:
                    image = self.signVid.showText(image, isDark=False, isHandVisible=False)
                else:
                    errFunc("Hand Not Visible.")

        if showFPS:
            image = self.signVid.showFPS(image) # Adding FPS to the Image

        return image

    def getHandSignPoints(self):
        visibility = False
        handIndex = [np.zeros(21*3), np.zeros(21*3)]

        if self.signVid.results and self.signVid.results.multi_hand_landmarks:
            for idx, hand_landmark in enumerate(self.signVid.results.multi_hand_landmarks):
                label = self.signVid.results.multi_handedness[idx].classification[0].label
                points = self.signVid.getHandPoints(hand_landmark)

                handIndex[label == "Left" and 0 or 1] = points
                visibility = points.any()

            if(visibility):
                self.signVid.keyPoints.append(np.array(handIndex).flatten())
                self.signVid.keyPoints = self.signVid.keyPoints[(self.signVid.seqLength * -1):]
        
        return(visibility)

    def getSignPrediction(self, callBack=None):
        if len(self.signVid.keyPoints) == self.signVid.seqLength:
            predict = self.model.predict(np.expand_dims(self.signVid.keyPoints, axis=0))[0]
            maxPredict = np.argmax(predict)
            predictStr = self.actions[maxPredict]

            if predict[maxPredict] > 0.8 and predictStr != self.signVid.lastPredict:
                    self.signVid.lastPredict = predictStr
                    if callBack != None:
                        callBack(predictStr)


def main():
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vc.set(cv2.CAP_PROP_FPS, 30)
    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    tracker = handSign()

    while vc.isOpened():
        success, frame = vc.read()
        image = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect

        image = tracker.startSignPrediction(image, showFPS=True)
        
        cv2.imshow("Hand Gesture Detection", image)
        
        if cv2.waitKey(1) == 27: # exit on ESC
            break
    
    vc.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()