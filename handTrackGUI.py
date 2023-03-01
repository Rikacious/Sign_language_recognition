import cv2
from threading import Thread
from handTrack import handTracker
from GUI import TrackerGUI


class handTrackerGUI(handTracker, TrackerGUI):
    def __init__(self):
        super().__init__(maxHands=2, detectionCon=0.6)

    def start(self, frame):
        frame = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect
        frameVisible = self.checkFrameVisibility(frame) # Checking Frame Visibility

        print(self.getPredictOption)

        if not frameVisible:
            image = self.showTextOnScreen(frame, isDark=True)
        else:
            self.getHandPosition(frame) # Track Hand Position with MediaPipe
        #     handVisible = self.getHandVisibility() # Checking Hand Visibility

        #     if(handVisible):
        #         image = self.handsFinder(frame) # Showing Hand Links in the Frame
        #         Thread(target=self.getPrediction).start() # Starting Prediction in Annother Thread
        #         # self.getPrediction() # Starting Prediction
        #         # image = self.showTextOnScreen(image, output=True)
        #     else:
        #         image = self.showTextOnScreen(frame, isDark=False, isHandVisible=False)

        image = self.showFPS(frame) # Adding FPS to the Image
        return image



if __name__ == "__main__":
    trackGUI = handTrackerGUI()
    trackGUI.initiate()
    trackGUI.showFrame(callBack=trackGUI.start)
    trackGUI.mainloop()