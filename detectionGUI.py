import cv2
from threading import Thread

from GUI import TrackerGUI
from handNumber import handNumber
from handAlphabet import handAlphabet
from handSignLanguage import handSign


class handDetectGUI(handNumber, handAlphabet, handSign, TrackerGUI):
    def __init__(self):
        super().__init__()

    def start(self, image):
        if self.getPredictOption == 0:
            self.startNumPrediction(image, errFunc=self.updateWarn, opFunc=self.writeOutput)
        elif self.getPredictOption == 1:
            self.startAlpPrediction(image, errFunc=self.updateWarn, opFunc=self.writeOutput)
        elif self.getPredictOption == 2:
            self.startSignPrediction(image, errFunc=self.updateWarn, opFunc=self.writeOutput)

        return image



if __name__ == "__main__":
    trackGUI = handDetectGUI()
    trackGUI.initiate()
    trackGUI.showFrame(callBack=trackGUI.start)
    trackGUI.mainloop()