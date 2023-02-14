import cv2
import mediapipe as mp
import numpy as np
import time


class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, modelComplexity=1, trackCon=0.5, dimension=(0,0,0)):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.dimension = dimension
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def placeRectangle(self, image):
        gap = 0.1
        thickness = 3
        color = (255, 0, 0)
        self.boxStart = (int(self.dimension[1] * gap), int(self.dimension[0] * gap))
        self.boxEnd = (int(self.dimension[1] * (1-gap)), int(self.dimension[0] * (1-gap)))
        # Placing the Rectangle
        cv2.rectangle(image, self.boxStart, self.boxEnd, color, thickness)

        return image


    def showTextOnScreen(self, image, found = True):
        fontScale = 1
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = found and (0, 255, 0) or (0, 0, 255)

        text = found and 'Hands Detected Properly' or 'Hands Not Detected Properly'
        textSize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = int((self.dimension[1] - textSize[0]) / 2)
        textY = int((self.boxStart[1] + textSize[1]) / 2)

        cv2.putText(image, text, (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA)

        if not found:
            hintFontScale = 0.8
            hintThickness = 1
            hint = 'Place the hands inside the Blue Rectangle'

            hintSize = cv2.getTextSize(hint, font, hintFontScale, hintThickness)[0]
            textX = int((self.dimension[1] - hintSize[0]) / 2)
            hintY = int(((self.boxStart[1] + hintSize[1]) / 2) + self.boxEnd[1])

            cv2.putText(image, hint, (textX, hintY), font, hintFontScale, color, hintThickness, cv2.LINE_AA)

        return image


    def checkImgDark(self, image):
        blur = cv2.blur(image, (5,5))
        mean = np.mean(blur)
        return(mean > 108 and True or False)


    def handsFinder(self, image, found = False):
        if found and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)

        image = self.showTextOnScreen(image, found)

        return image


    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(Hand.landmark):
                h,w,c = self.dimension
                cx,cy = int(lm.x * w), int(lm.y * h)
                sx, sy = self.boxStart
                ex, ey = self.boxEnd
                if((cx > sx and cx < ex) and (cy > sy and cy < ey)):
                    lmlist.append([id, cx, cy])
                else:
                    return []
            
            # if draw:
                # cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)

        return lmlist



def main():
    vc = cv2.VideoCapture(0)

    currentTime = 0
    previousTime = 0

    if vc.isOpened(): # try to get the first frame
        success, image = vc.read()
        tracker = handTracker(maxHands=1, dimension=image.shape)
    else:
        success = False

    while success:
        if(tracker.checkImgDark(image)):
            print('LIGHT')
        else:
            print('DARK')

        image = tracker.placeRectangle(image)
        lmList = tracker.positionFinder(image)

        image = tracker.handsFinder(image, (len(lmList) > 0))

        # Adding FPS to the Image
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        # Displaying FPS on the image
        cv2.putText(image, str(int(fps))+" FPS", (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,0,255), 1)

        cv2.imshow("Hand Gesture Detection", image)
        success, image = vc.read()
        
        if cv2.waitKey(10) == 27: # exit on ESC
            break
    
    vc.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()