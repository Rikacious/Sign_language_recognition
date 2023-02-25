import os
import cv2
import json
import time
import numpy as np
import mediapipe as mp
from threading import Thread


class dataCollection:
    def __init__(self, mode=False, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        jsonFile = open("settings.json")
        settings = json.load(jsonFile)
        jsonFile.close()

        self.motionKeyPoints = []
        self.noSequence = settings['noSequence']
        self.sequenceLength = settings['sequenceLength']
        self.DATA_FOLDER = os.path.join(settings['rawDataDir'])
        self.createActions = np.array(settings['collectActions'])
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode, 2, modelComplexity, detectionCon, trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    @property
    def settingsParameters(self):
        return self.createActions, self.noSequence, self.sequenceLength

    def createDirectories(self):
        for action in self.createActions:
            print(f"Creating Directory for {action}")
            try:
                os.makedirs(os.path.join(self.DATA_FOLDER, action))
            except:
                pass
            print(f"Directory Created for {action}")        

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

    def getHandPosition(self, image):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

    def handsFinder(self, image):
        if self.results.multi_hand_landmarks:
            for handLimbs in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(image, handLimbs, self.mpHands.HAND_CONNECTIONS)

        return image

    def getKeyPoints(self):
        threshold = 0.015
        keyPoints = [np.zeros(21*3), np.zeros(21*3)]

        if self.results.multi_hand_landmarks:
            if(len(self.results.multi_handedness) > 1):
                for idx, hand_landmark in enumerate(self.results.multi_hand_landmarks):
                    points = np.array([
                        [res.x, res.y, res.z] for res in hand_landmark.landmark
                    ])

                    x0, y0, _ = points[0]
                    x5, y5, _ = points[5]
                    x17, y17, _ = points[17]

                    if(abs(x0*(y5-y17) + x5*(y17-y0) + x17*(y0-y5)) > threshold):
                        keyPoints[idx] = points.flatten()
            else:
                points = np.array([
                    [res.x, res.y, res.z] for res in self.results.multi_hand_landmarks[0].landmark
                ])

                x0, y0, _ = points[0]
                x5, y5, _ = points[5]
                x17, y17, _ = points[17]

                if(abs(x0*(y5-y17) + x5*(y17-y0) + x17*(y0-y5)) > threshold):
                    label = self.results.multi_handedness[0].classification[0].label
                    keyPoints[label == "Left" and 0 or 1] = points.flatten()
        
            self.motionKeyPoints.append(np.array(keyPoints).flatten())

    def storeKeyPoints(self, action, sequence):
        nPyPath = os.path.join(self.DATA_FOLDER, action, str(sequence))
        np.save(nPyPath, np.array(self.motionKeyPoints))
        self.motionKeyPoints = []

    def updateSettings(self):
        jsonFile = open('settings.json', 'r+')
        settings = json.load(jsonFile)
        settings['actions'].extend(settings['collectActions'])
        settings['collectActions'] = []
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

    if vc.isOpened():
        collect = dataCollection(detectionCon=0.6)
        Thread(target=collect.createDirectories).start()
        # collect.createDirectories()
        # Collecting Parameters Getting
        actions, noSequence, sequenceLength = collect.settingsParameters

        for action in actions:
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
                
                for frameNum in range(sequenceLength):
                    success, frame = vc.read()
                    frame = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect

                    collect.getHandPosition(frame) # Track Hand Position with MediaPipe
                    image = collect.handsFinder(frame) # Showing Hand Links in the Frame

                    if frameNum != 0:
                        print(f"{action} || Video: {sequence} || Frame: {frameNum}")
                    
                    image = collect.showWaitingText(image, f"{action} || VIDEO: {sequence}")
                    cv2.imshow("Collecting Hand Data", image)
                    collect.getKeyPoints()
        
                    if cv2.waitKey(10) == 27: # exit on ESC
                        break

                collect.storeKeyPoints(action, sequence)
                

        collect.updateSettings()
        
        vc.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()