import cv2
import mediapipe as mp
import numpy as np
import time
import math


class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.currentTime = 0
        self.previousTime = 0
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def checkFrameVisibility(self, image):
        blur = cv2.blur(image, (5,5))
        mean = np.mean(blur)
        return(mean > 80 and True or False)

    def getHandPosition(self, image, handNo=0):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        self.lmList = []

        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])

    def getHandVisibility(self):
        visibility = 0

        if len(self.lmList) > 0:
            _, x5, y5 = self.lmList[5]
            _, x17, y17 = self.lmList[17]

            visibility = math.sqrt((x17-x5)**2 + (y17-y5)**2)
            # print(visibility)
        
        return(visibility > 55 and True or False)

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

        # textX = int((image.shape[1] - hintSize[0]) / 2)
        # hintY = int(((boxStart[1] + hintSize[1]) / 2) + int(image.shape[0] * (1-gap))

        return image

    def handsFinder(self, image):
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)

        return image

    def showFPS(self, image):
        self.currentTime = time.time()
        fps = 1 / (self.currentTime - self.previousTime)
        self.previousTime = self.currentTime
        # Displaying FPS on the image
        cv2.putText(image, str(int(fps))+" FPS", (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,0,255), 1)

        return image


def main():
    vc = cv2.VideoCapture(0)
    tracker = handTracker(maxHands=1)

    while vc.isOpened():
        success, frame = vc.read()
        frameVisible = tracker.checkFrameVisibility(frame)

        if not frameVisible:
            image = tracker.showTextOnScreen(frame, isDark=True)
        else:
            tracker.getHandPosition(frame)
            handVisible = tracker.getHandVisibility()

            if(handVisible):
                image = tracker.handsFinder(frame)
                image = tracker.showTextOnScreen(image, isDark=False, isHandVisible=True)
            else:
                image = tracker.showTextOnScreen(frame, isDark=False, isHandVisible=False)

        image = tracker.showFPS(image) # Adding FPS to the Image
        cv2.imshow("Hand Gesture Detection", image)
        success, image = vc.read()
        
        if cv2.waitKey(10) == 27: # exit on ESC
            break
    
    vc.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()