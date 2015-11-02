#Basic utilities for saving and loading data

import pickle

def saveData(data, outputDest):
    dataOutput = open(outputDest, 'wb')
    pickle.dump(data, dataOutput)
    dataOutput.close()

def loadData(inputSource):
    dataInput = open(inputSource, 'rb')
    data = pickle.load(dataInput)
    dataInput.close()
    return data
