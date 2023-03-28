import cv2
import mediapipe as mp
import time
import math
import numpy as np

from video import Video
 
 
class handDetector():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tipIds = [4, 8, 12, 16, 20]
        self.trkVid = Video(hands=1, detectionCon=0.6, seqLength=3)
 
    def startHandTracking(self, image, errFunc=None, opFunc=None, showFPS=False, draw=True):
        if not self.trkVid.checkVisibility(image): # Checking Frame Visibility
            if errFunc == None:
                image = self.trkVid.showText(image, isDark=True)
            else:
                errFunc("Video too Dark.")
        else:
            self.trkVid.getHandPosition(image) # Track Hand Position with MediaPipe

            if(self.getHandNumPoints()):
                image = self.trkVid.handsFinder(image) # Showing Hand Links in the Frame

                if opFunc == None:
                    image = self.trkVid.showText(image, output=True)
            else:
                if errFunc == None:
                    image = self.trkVid.showText(image, isDark=False, isHandVisible=False)
                else:
                    errFunc("Hand Not Visible.")

        if showFPS:
            image = self.trkVid.showFPS(image) # Adding FPS to the Image

        return image, self.lmList

    def getHandNumPoints(self):
        self.lmList = []

        if self.trkVid.results and self.trkVid.results.multi_hand_landmarks:
            hand_landmark = self.trkVid.results.multi_hand_landmarks[0]
            points = self.trkVid.getHandPoints(hand_landmark)

            for id, point in enumerate(points):
                if isinstance(point, np.float64):
                    continue
                x, y = int(point[0] * 640), int(point[1] * 480)
                self.lmList.append((id, x, y))

            self.trkVid.keyPoints.append(np.array(self.lmList))
            self.trkVid.keyPoints = self.trkVid.keyPoints[(self.trkVid.seqLength * -1):]
        
        return(np.array(self.lmList).any())

    def fingersUp(self):
        fingers = [0, 0, 0, 0, 0]

        if len(self.lmList) > 0:
            # Thumb
            fingers[0] = (self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]) and 1 or 0
            # Fingers
            for id in range(1, 5):
                fingers[id] =  (self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]) and 1 or 0
 
        return fingers
 
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
 
        return length, img, [x1, y1, x2, y2, cx, cy]



def main():
    ##########################
    wCam, hCam = 640, 480
    frameR = 100 # Frame Reduction
    smoothening = 7
    #########################
    
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vc.set(cv2.CAP_PROP_FPS, 30)
    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    detector = handDetector()
    wScr, hScr = (1920, 1080)
    # print(wScr, hScr)
    
    while True:
        # 1. Find hand Landmarks
        success, img = vc.read()
        img = cv2.flip(img, 1) # Flipping Frame to get Mirror Effect
        img, lmList = detector.startHandTracking(img, showFPS=True)
        
        # 2. Get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            # print(x1, y1, x2, y2)
        
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
        
            # 7. Move Mouse
            # autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
            detector.trkVid.lastPredict = f"Cursor at: {str(int(wScr - clocX))}, {str(int(clocY))}"
            
        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            # 10. Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                15, (0, 255, 0), cv2.FILLED)
                # autopy.mouse.click()
                detector.trkVid.lastPredict = "Mouse Click"
        
        # 12. Display
        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()