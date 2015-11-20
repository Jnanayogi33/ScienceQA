#Basic utilities for saving and loading and downloading data, creating pool of workers for distributed tasks
import pickle, time, re
from queue import Queue
from threading import Thread


# Takes input from inputQueue, applies work function, puts return into outputQueue
#  - workFunction must have only one input parameter, one output
#  - Only one process should be adding items into the inputQueue or taking them out of outputQueue
#  - Centralize resource access to remain thread-safe
class Worker(Thread):
    def __init__(self, inputQueue, outputQueue, workFunction):
        Thread.__init__(self)
        self.inputQueue = inputQueue
        self.outputQueue = outputQueue
        self.workFunction = workFunction
    def run(self):
        while True:
            task = self.inputQueue.get()
            self.outputQueue.put(self.workFunction(task))
            self.inputQueue.task_done()


# Creates pool of workers to complete a task
#  - Each worker uses workFunction to accomplish one task using one item from inputList
#  - Outputs placed in an outputQueue which is then converted into a list and returned
#  - Work function should be built so that in case of failure/exception they return "None" or else all processes will be stopped
#  - Pool will delete the None results and just return properly formatted results
#  - Does not maintain order of inputs/outputs
#  - Default of 20 workers because for API downloading that is max in China that doesn't seem to be blocked
def workerPool(inputList, workFunction, numWorkers=20):
    inputQueue = Queue()
    for inputItem in inputList:
        inputQueue.put(inputItem)
    outputQueue = Queue()
    for _ in range(numWorkers):
        worker = Worker(inputQueue, outputQueue, workFunction)
        worker.daemon = True
        worker.start()
    inputQueue.join()
    outputList = []
    lossCount = 0
    while not outputQueue.empty():
        elem = outputQueue.get()
        if elem is not None: outputList += [elem]
        else: lossCount += 1 
        outputQueue.task_done()
    outputQueue.join()
    print("Worker pool improper function return resulting loss count:", lossCount)
    return outputList

# Basic save data function
def saveData(data, outputDest):
    dataOutput = open(outputDest, 'wb')
    pickle.dump(data, dataOutput)
    dataOutput.close()

# Basic load data function
def loadData(inputSource):
    dataInput = open(inputSource, 'rb')
    data = pickle.load(dataInput)
    dataInput.close()
    return data
