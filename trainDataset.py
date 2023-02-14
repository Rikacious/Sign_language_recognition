import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


def main():
    DATA_PATH = os.path.join('MP_DATA')
    noSequence = 30
    sequenceLength = 10
    actions = np.array(["ONE", "TWO", "THREE"])
    labelMap = {label:num for num, label in enumerate(actions)}

    sequences, labels = [], []
    for action in actions:
        for sequence in range(noSequence):
            window = []
            for frameNum in range(sequenceLength):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frameNum)))
                window.append(res)
            sequences.append(window)
            labels.append(labelMap[action])

    sequenceArr = np.array(sequences)
    labesArr = to_categorical(labels).astype(int)

    seqTrain, seqTest, lblTrain, lblTest = train_test_split(sequenceArr, labesArr, test_size=0.05)

    logDir = os.path.join('Logs')
    tbCallback = TensorBoard(log_dir=logDir)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,63)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(seqTrain, lblTrain, epochs=200, callbacks=[tbCallback])
    model.summary()

    res = model.predict(seqTest)

    for i in range(5):
        print("Predict: {} || Actual: {}".format(actions[np.argmax(res[i])], actions[np.argmax(lblTest[i])]))

    model.save('action.h5')



    



if __name__ == "__main__":
    main()