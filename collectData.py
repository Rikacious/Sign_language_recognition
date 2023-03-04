import os
import cv2
import json
import time
import numpy as np
import mediapipe as mp
from threading import Thread

from video import Video


class dataCollection(Video):
    def __init__(self, hands=2, detectionCon=0.5, seqLength=0, type=None, **kwargs):
        super().__init__(**kwargs)
        jsonFile = open("settings.json")
        settings = json.load(jsonFile)
        jsonFile.close()

        self.type = type
        self.noSequence = settings['noSequence']
        self.DATA_FOLDER = os.path.join(settings['rawDataDir'], str(type.upper()))
        self.createActions = np.array(settings['collectActions'])
    
    @property
    def settingsParameters(self):
        return self.createActions, self.noSequence, self.seqLength

    def createDirectory(self, action):
        try:
            os.makedirs(os.path.join(self.DATA_FOLDER, action))
            print(f"Directory Created for {action}")        
        except:
            pass

    def showWaitingText(self, image, text="", timer=0):
        gap = 0.1
        thickness = 1
        fontScale = 0.7
        color = (0,0,255)
        textPos = (10, 25)
        font = cv2.FONT_HERSHEY_SIMPLEX

        timeLeft = timer - time.time()

        if timeLeft > 1:
            timerText = text + " || " + str(int(timeLeft))
        elif timeLeft > 0 and timeLeft < 1:
            timerText = text + " || " + "Starting"
        else:
            timerText = text + " || " + "Capturing"

        cv2.putText(image, timerText, textPos, font, fontScale, color, thickness, cv2.LINE_AA)
        return image

    def getKeyPoints(self):

        if self.results.multi_hand_landmarks:
            if self.type == "sign":
                keyPoints = [np.zeros(21*3), np.zeros(21*3)]

                if(len(self.results.multi_handedness) > 1):
                    for idx, hand_landmark in enumerate(self.results.multi_hand_landmarks):
                        points = self.getHandPoints(hand_landmark)
                        handPoints.append(points)
                else:
                    hand_landmark = self.results.multi_hand_landmarks[0]
                    label = self.results.multi_handedness[0].classification[0].label
                    points = self.getHandPoints(hand_landmark)
                    keyPoints[label == "Left" and 0 or 1] = points
            
                self.keyPoints.append(np.array(keyPoints).flatten())

            elif self.type == "char":
                hand_landmark = self.results.multi_hand_landmarks[0]
                points = self.getHandPoints(hand_landmark)
                self.keyPoints.append(points)

    def storeKeyPoints(self, action, sequence):
        nPyPath = os.path.join(self.DATA_FOLDER, action, str(sequence))
        np.save(nPyPath, np.array(self.keyPoints))
        self.keyPoints = []

    def updateSettings(self, action):
        jsonFile = open('settings.json', 'r+')
        settings = json.load(jsonFile)

        settings['actions'][self.type].append(action)
        settings['collectActions'].remove(action)
        jsonFile.seek(0)
        jsonFile.truncate()
        json.dump(settings, jsonFile, indent=4)
        jsonFile.close()


def main():
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vc.set(cv2.CAP_PROP_FPS, 30)
    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    collectType = "char" # char | sign

    if vc.isOpened():
        if collectType == "char":
            collect = dataCollection(hands=1, detectionCon=0.6, seqLength=10, type="char")
        elif collectType == "sign":
            collect = dataCollection(hands=2, detectionCon=0.6, seqLength=20, type="sign")
        else:
            return

        # Collecting Parameters Getting
        actions, noSequence, seqLength = collect.settingsParameters

        for action in actions:
            collect.createDirectory(action) # Creating Folder for Action

            for sequence in range(noSequence):
                timerEnd = time.time() + 3 + 0.8 # Seeting Timer for Break between Videos
                print(f"{action} || Video: {sequence} || Frame: 0 || WAITING")
                while(timerEnd - time.time() > -0.2): # Loop until timer goes 0. 0.2 for Capture Overlap
                    success, frame = vc.read()
                    frame = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect

                    collect.getHandPosition(frame) # Track Hand Position with MediaPipe
                    image = collect.handsFinder(frame) # Showing Hand Links in the Frame
                    image = collect.showWaitingText(image, f"{action} || VIDEO: {sequence}", timerEnd)
                    cv2.imshow("Collecting Hand Data", image)
                    cv2.waitKey(10)
                
                frameNum = 0
                while frameNum < seqLength:
                    success, frame = vc.read()
                    frame = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect

                    collect.getHandPosition(frame) # Track Hand Position with MediaPipe
                    image = collect.handsFinder(frame) # Showing Hand Links in the Frame

                    if frameNum != 0:
                        print(f"{action} || Video: {sequence} || Frame: {frameNum}")
                    
                    image = collect.showWaitingText(image, f"{action} || VIDEO: {sequence}")
                    cv2.imshow("Collecting Hand Data", image)
                    collect.getKeyPoints()
                    frameNum += 1
        
                    if cv2.waitKey(10) == 27: # exit on ESC
                        break

                collect.storeKeyPoints(action, sequence)

            collect.updateSettings(action) # Updating Specific Action to Complete
        
        vc.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()