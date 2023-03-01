import cv2
import time
import tkinter as TK
from PIL import Image, ImageTk
from threading import Thread
import customtkinter as CTKinter

CTKinter.set_appearance_mode("System")  # Modes: system (default), light, dark
CTKinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

class TrackerGUI(CTKinter.CTk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.currentTime = 0 
        self.previousTime = 0
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

        self.vidCanvas = CTKinter.CTkCanvas(self, width=640, height=480, highlightthickness=0, bg="black")
        self.vidCanvas.pack(padx=10, pady=10)

        self.vidFrame = self.vidCanvas.create_image(0,0, anchor=TK.NW)
        self.vidFPS = self.vidCanvas.create_text(0,-4, anchor=TK.NW, font=("Consolas", 18), fill="red")
        self.vidWarn = self.vidCanvas.create_text(320,465, anchor=TK.CENTER, text="Some Warn", font=("Consolas", 22), fill="red")

        self.optionBtn = CTKinter.CTkSegmentedButton(self, values=self.predictTypes, corner_radius=4, border_width=1, command=self.changeType)
        self.optionBtn.set(self.predictTypes[self.predictOption])
        self.vidCanvas.create_window(640, 0, anchor=TK.NE, window=self.optionBtn)

        self.opFrame = CTKinter.CTkTextbox(self, width=640, height=280, state="disabled", font=("", 18), spacing2=6, wrap=TK.WORD)
        self.opFrame.pack(padx=10, pady=(0, 10))

    def showVidFPS(self):
        self.currentTime = time.time()
        fps = 1 / (self.currentTime - self.previousTime)
        self.previousTime = self.currentTime
        self.vidCanvas.itemconfig(self.vidFPS, text=str(f"FPS:{int(fps)}"))

    def showFrame(self, callBack=None):
        global imgCTK
        _, frame = self.vc.read()
        frame = cv2.flip(frame, 1) # Flipping Frame to get Mirror Effect

        if callBack != None:
            frame = callBack(frame)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        prevImg = Image.fromarray(cv2image)
        imgCTK = ImageTk.PhotoImage(image=prevImg)

        self.vidCanvas.itemconfig(self.vidFrame, image=imgCTK)
        self.showVidFPS()
        Thread(
            target= self.vidCanvas.after, 
            args= (1, lambda: self.showFrame(callBack))
        ).start()
        
    def changeType(self, value):
        self.predictOption = self.predictTypes.index(value)
        # self.writeOutput(value)
        self.updateWarn(value)

    def updateWarn(self, text=""):
        self.vidCanvas.itemconfig(self.vidWarn, text=str(text))

    def writeOutput(self, text=None):
        if text:
            self.opFrame.configure(state="normal")
            self.opFrame.insert(TK.END, text + " ")
            self.opFrame.see(TK.END)
            self.opFrame.configure(state="disabled")
    


if __name__ == "__main__":
    app = TrackerGUI()
    app.initiate()
    app.showFrame()
    app.mainloop()