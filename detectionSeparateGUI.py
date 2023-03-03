import cv2
from threading import Thread

from GUI import TrackerGUI
from handCharacter import handCharacter
from handSign import handSign


class handDetectGUI(handCharacter, handSign, TrackerGUI):
    def __init__(self):
        super().__init__()

    def start(self, image):
        if self.getPredictOption in [0,1]:
            self.startCharPrediction(image, errFunc=self.updateWarn, opFunc=self.writeOutput)
        else:
            self.startSignPrediction(image, errFunc=self.updateWarn, opFunc=self.writeOutput)

        return image



if __name__ == "__main__":
    trackGUI = handDetectGUI()
    trackGUI.initiate()
    trackGUI.showFrame(callBack=trackGUI.start)
    trackGUI.mainloop()