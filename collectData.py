import os
import cv2
import json
import numpy as np
import mediapipe as mp
from threading import Thread


class dataCollection:
    def __init__(self, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        jsonFile = open('settings.json')
        settings = json.load(jsonFile)
        jsonFile.close()
        
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.noSequence = settings['noSequence']
        self.sequenceLength = settings['sequenceLength']
        self.DATA_FOLDER = os.path.join(settings['rawDataDir'])
        self.createActions = np.array(settings['collectActions'])
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(False, 2,self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    @property
    def settingsParameters(self):
        return self.createActions, self.noSequence, self.sequenceLength

    def createDirectories(self):
        for action in self.createActions:
            print(f"Creating Directory for {action}")
            for sequence in range(self.noSequence):
                try:
                    os.makedirs(os.path.join(self.DATA_FOLDER, action, str(sequence)))
                except:
                    pass
            print(f"Directory Created for {action}")        

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

        npyPath = os.path.join(self.DATA_FOLDER, action, str(sequence), str(frameNum))
        np.save(npyPath, keyPoints)

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
                print(f"{action} || Video: {sequence} || Frame: 0 || WAITING")
                for i in range(70):
                    success, frame = vc.read()
                    frame = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect

                    collect.getHandPosition(frame) # Track Hand Position with MediaPipe
                    image = collect.handsFinder(frame) # Showing Hand Links in the Frame

                    cv2.imshow("Collecting Hand Data", image)
                    cv2.waitKey(10)

                for frameNum in range(sequenceLength):
                    success, frame = vc.read()
                    frame = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect

                    collect.getHandPosition(frame) # Track Hand Position with MediaPipe
                    image = collect.handsFinder(frame) # Showing Hand Links in the Frame

                    if frameNum != 0:
                        print(f"{action} || Video: {sequence} || Frame: {frameNum}")
                    
                    cv2.imshow("Collecting Hand Data", image)
                    collect.storeKeyPoints(action, sequence, frameNum)
        
                    if cv2.waitKey(10) == 27: # exit on ESC
                        break

        collect.updateSettings()
        
        vc.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()