import cv2
from threading import Thread
from handTrack import handTracker
from GUI import TrackerGUI


class handDetectGUI(handTracker, TrackerGUI):
    def __init__(self):
        super().__init__(maxHands=2, detectionCon=0.6)

    def start(self, image):
        frameVisible = self.checkFrameVisibility(image) # Checking Frame Visibility

        if not frameVisible:
            self.updateWarn("Frame Not Visible")
        else:
            Thread(
                target=lambda: self.getHandPosition(image=image),
            ).start() # Starting Prediction in Annother Thread
            # self.getHandPosition(image) # Track Hand Position with MediaPipe
            handVisible = self.getHandVisibility() # Checking Hand Visibility

            if(handVisible):
                self.updateWarn()
                # image = self.handsFinder(image) # Showing Hand Links in the Frame
                Thread(
                    target=lambda: self.getPrediction(callBack=self.writeOutput),
                ).start() # Starting Prediction in Annother Thread
            else:
                self.updateWarn("Hand Not Visible")

        return image



if __name__ == "__main__":
    trackGUI = handDetectGUI()
    trackGUI.initiate()
    trackGUI.showFrame(callBack=trackGUI.start)
    trackGUI.mainloop()