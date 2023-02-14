import os
import json
import uuid
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import models
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard



class TrainDataset:
    def __init__(self, testSize=0.05):
        jsonFile = open('settings.json')
        settings = json.load(jsonFile)
        jsonFile.close()

        self.testSize = testSize
        self.noSequence = settings['noSequence']
        self.sequenceLength = settings['sequenceLength']
        self.DATA_FOLDER = os.path.join(settings['rawDataDir'])
        self.logDir = os.path.join(settings['logDir'])
        self.modelsDir = os.path.join(settings['modelsDir'])
        self.lastModel = settings['lastModel']
        self.actions = np.array(settings['actions'])

    def getStoredData(self):
        sequences, labels = [], []
        labelMap = {label:num for num, label in enumerate(self.actions)}

        for action in self.actions:
            for sequence in range(self.noSequence):
                window = []
                for frameNum in range(self.sequenceLength):
                    res = np.load(os.path.join(self.DATA_FOLDER, action, str(sequence), "{}.npy".format(frameNum)))
                    window.append(res)
                sequences.append(window)
                labels.append(labelMap[action])

        sequenceArr = np.array(sequences)
        labelsArr = to_categorical(labels).astype(int)

        seqTrain, seqTest, lblTrain, lblTest = train_test_split(sequenceArr, labelsArr, test_size=0.05)

        self.sequence = (seqTrain, seqTest)
        self.labels = (lblTrain, lblTest)

    def loadStoredModel(self):
        if self.lastModel != "":
            self.model = models.load_model(os.path.join(self.modelsDir, self.lastModel))
        else:
            print("No Last Model Found.")

    def startTrainning(self):
        tbCallback = TensorBoard(log_dir=self.logDir)
        seqTrain = self.sequence[0]
        lblTrain = self.labels[0]

        self.model = models.Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,63)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))

        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.fit(seqTrain, lblTrain, epochs=200, callbacks=[tbCallback])
        self.model.summary()

    def predictData(self):
        seqTest = self.sequence[1]
        lblTest = self.labels[1]
        count = 0

        prediction = self.model.predict(seqTest)

        for i in range(len(seqTest)):
            predictData = self.actions[np.argmax(prediction[i])]
            labeledData = self.actions[np.argmax(lblTest[i])]
            print(f"Predict: {predictData} || Actual: {labeledData}")
            if predictData == labeledData:
                count = count + 1

        print(f"Total Accuracy: {(count/len(seqTest))*100}")

    def storeTrainedModel(self):
        modelName = str(uuid.uuid1()) + '.h5'
        self.model.save(os.path.join(self.modelsDir, modelName))

        jsonFile = open('settings.json', 'r+')
        settings = json.load(jsonFile)
        settings['lastModel'] = modelName
        jsonFile.seek(0)
        json.dump(settings, jsonFile, indent=4)
        jsonFile.close()


def main():
    trainModel = True

    train = TrainDataset()
    train.loadStoredModel()
    train.getStoredData()

    if trainModel:
        train.startTrainning()
        train.predictData()
        train.storeTrainedModel()
    else:
        train.predictData()


if __name__ == "__main__":
    main()