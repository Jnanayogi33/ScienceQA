#Basic utilities for saving and loading and downloading data, creating pool of workers for distributed tasks
import pickle, time, re, sys, os
from queue import Queue
import queue
from threading import Thread


# Takes input from inputQueue, applies work function, puts return into outputQueue
#  - If any exception caught in process, puts a "None" into outputQueue
#  - Otherwise puts the tuple (task, result) into outputQueue
#  - workFunction must have only one input parameter, one output
class Worker(Thread):
    def __init__(self, workerID, inputQueue, outputQueue, workFunction):
        Thread.__init__(self)
        self.workerID = workerID
        self.inputQueue = inputQueue
        self.outputQueue = outputQueue
        self.workFunction = workFunction
    def run(self):
        while self.inputQueue.qsize() > 0:
            print(self.workerID, "-- Tasks left in input queue:", self.inputQueue.qsize())
            task = self.inputQueue.get()
            try:
                result = self.workFunction(task)
                self.outputQueue.put((task, result))
                self.inputQueue.task_done()
            except:
                e = sys.exc_info()[0]
                print("Caught exception", e)
                self.outputQueue.put(None)
                self.inputQueue.task_done()
                pass


# Creates pool of workers to complete a task
#  - Each worker uses workFunction to accomplish one task using one item from inputList
#  - Outputs in form (task, result) placed in an outputQueue which is then converted into a list and returned
#  - If certain function call fails, expects worker to return None
#  - Pool will try those that return None iterations times, if still doesn't work, then will place (task, None) in output list
#  - Refresh one worker every 60 seconds because on larger numbers of tasks eventually most of the threads get stuck somewhere
def workerPool(inputList, workFunction, numWorkers=20, iterations=3, redundancies=False):   
    outputList = []
    for iteration in range(iterations):
        inputQueue = Queue()
        for inputItem in inputList:
            inputQueue.put(inputItem)
        outputQueue = Queue()
        workers = []
        for i in range(numWorkers):
            worker = Worker(i, inputQueue, outputQueue, workFunction)
            worker.daemon = True
            workers += [worker]
            worker.start()
        if redundancies != True:
            inputQueue.join()
        else:
            while inputQueue.qsize() > 0:
                for i in range(numWorkers):
                    if inputQueue.qsize() == 0: break
                    if workers[i].is_alive():
                        workers[i].join(30)
                        workers[i] = Worker(i, inputQueue, outputQueue, workFunction)
                        workers[i].daemon = True
                        workers[i].start()
                        print("Refreshed worker", i)
                        time.sleep(30)
            time.sleep(30)
            for worker in workers:
                if worker.is_alive():
                    print("Joining worker for this round:", worker.workerID) 
                    worker.join(1)
        print("Worker Pool: Done joining input queue for iteration round", iteration+1)
        while not outputQueue.empty():
            elem = outputQueue.get()
            if elem is not None: 
                outputList += [elem]
                inputList.remove(elem[0])
            outputQueue.task_done()
        outputQueue.join()
    for inputItem in inputList:
        outputList += [(inputItem, None)]
    return outputList


# Master function for downloading in chunks, saving progress in case of download failure
#  - Downloads in chunks of 1000 items at a time
#  - Saves each chunk into cache, then combines at end
#  - By default runs one last iteration to see if it can catch ones that returned None, can run more if change default
#  - Returns results in form of one combined dictionary
#  - Saves everything to deal with fact that connection breaks are frequent, but after every file done downloading, cleans up cache
#  - Remember that list(set()) operation DOES NOT MAINTAIN ORDER! So if reload, use previous saved list
def poolDownloader(inputList, workerFunction, workerNum = 20, iterations=3, redundancies=False, poolRuns = 1):
    
    inputList = list(set(inputList))
    saveData(inputList, 'ScienceQASharedCache/inputList_poolRuns_' + str(poolRuns))
    # inputList = loadData('ScienceQASharedCache/inputList_poolRuns_' + str(poolRuns))
    
    folds = max(int(len(inputList)/1000),1)
    outputs = {}
    for i in list(range(0, folds)):
        lowSplit = int(i*len(inputList)/folds)
        highSplit = int((i+1)*len(inputList)/folds)
        print("Working on items", lowSplit, "to", highSplit, "out of total", len(inputList))
        rawList = workerPool(inputList[lowSplit:highSplit], workerFunction, workerNum, iterations, redundancies)
        currOutputs = {}
        for result in rawList: currOutputs[result[0]] = result[1]
        saveData(currOutputs, 'ScienceQASharedCache/currOutputs_' + str(i) + "_poolRuns_" + str(poolRuns))

    outputDict = {}
    for i in list(range(folds)):
        curr = loadData('ScienceQASharedCache/currOutputs_' + str(i) + "_poolRuns_" + str(poolRuns))
        for key in curr.keys():
            outputDict[key] = curr[key]
            if curr[key] != None and key in inputList: 
                inputList.remove(key)

    if poolRuns > 1:
        leftOvers = poolDownloader(inputList, workerFunction, workerNum, iterations, redundancies, poolRuns - 1)
        for key in leftOvers.keys():
            outputDict[key] = leftOvers[key]

    for i in list(range(folds)):
        os.remove('ScienceQASharedCache/currOutputs_' + str(i) + "_poolRuns_" + str(poolRuns))
    os.remove('ScienceQASharedCache/inputList_poolRuns_' + str(poolRuns))

    return outputDict




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
