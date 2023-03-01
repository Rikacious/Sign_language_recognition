import cv2
import time
import tkinter as TK
from PIL import Image
from threading import Thread
import customtkinter as CTKinter

CTKinter.set_appearance_mode("System")  # Modes: system (default), light, dark
CTKinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

class TrackerGUI(CTKinter.CTk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.predictOption = 0
        self.vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.predictTypes = ["CHARACTERS", "HAND GESTURES", "HAND MOVEMENT"]

        self.geometry("660x700")
        self.title("Hand Gesture Recognition")
        self.resizable(width=False, height=False)
        # self.bind('<Escape>', lambda e: self.quit())

    @property
    def getPredictOption(self):
        return self.predictOption
        
    @property
    def getPredictTypes(self):
        return self.predictTypes

    def initiate(self):
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.vc.set(cv2.CAP_PROP_FPS, 30)
        self.vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self.vidFrame = CTKinter.CTkLabel(self, compound=TK.CENTER, anchor=TK.CENTER, text="")
        self.vidFrame.configure(height=480, width=640)
        self.vidFrame.pack(padx=10, pady=10)

        self.optionBtn = CTKinter.CTkSegmentedButton(self, values=self.predictTypes, corner_radius=4, border_width=1, command=self.changeType)
        self.optionBtn.place(in_=self.vidFrame, relx=1, anchor=TK.NE)
        self.optionBtn.set(self.predictTypes[self.predictOption])

        self.opFrame = CTKinter.CTkTextbox(self, width=640, height=280, state="disabled", font=("", 18), spacing2=6, wrap=TK.WORD)
        self.opFrame.pack(padx=10, pady=(0, 10))

    def showFrame(self, callBack=None):
        _, frame = self.vc.read()

        if callBack != None:
            frame = callBack(frame)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        prevImg = Image.fromarray(cv2image)
        imgCTK = CTKinter.CTkImage(prevImg, size=(640,480))

        self.vidFrame.configure(image=imgCTK)
        Thread(
            target= self.vidFrame.after, 
            args= (1, lambda: self.showFrame(callBack))
        ).start()
        
    def changeType(self, value):
        self.predictOption = self.predictTypes.index(value)
        self.opFrame.configure(state="normal")
        self.opFrame.insert(TK.END, value + " ")
        self.opFrame.see(TK.END)
        self.opFrame.configure(state="disabled")


if __name__ == "__main__":
    app = TrackerGUI()
    app.initiate()
    app.showFrame()
    app.mainloop()