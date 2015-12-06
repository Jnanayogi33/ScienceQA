#Basic utilities for saving and loading and downloading data, creating pool of workers for distributed tasks
import pickle, time, re, sys, os
from queue import Queue
import queue
from threading import Thread
import numpy
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Store text data and index into it here
allTextLines = None
allTextVectorizer = None
allTextIndex = None
allTextAnalyzer = None

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


def getRelevantPassages(query, k):
    queryVector = allTextVectorizer.transform([query])
    queryIndices = numpy.array([allTextVectorizer.vocabulary_.get(word) for word in allTextAnalyzer(query)])
    queryIndices = [i for i in queryIndices if i is not None]
    querySimilarityScores = linear_kernel(queryVector[:,queryIndices], allTextIndex[:,queryIndices]).flatten()
    relatedDocIndices = querySimilarityScores.argsort()[:-k:-1]
    return [allTextLines[i] for i in relatedDocIndices]


# Save function for sparse matrices
def saveSparseCSR(array, filename):
    numpy.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


# Load function for sparse matrices
def loadSparseCSR(filename):
    if filename[-4:] != '.npz':
        filename = filename + '.npz'
    loader = numpy.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


# Convert wikipedia dictionary of pages into list of lines
def convertWikiPagesToLines(wikiKeyword2Pages):
    wikiLines = []
    for key in wikiKeyword2Pages:
        if wikiKeyword2Pages[key] == None: continue
        if wikiKeyword2Pages[key] == {}: continue
        for key2 in wikiKeyword2Pages[key]:
            if wikiKeyword2Pages[key][key2] == None: continue
            if wikiKeyword2Pages[key][key2] == []: continue
            wikiLines += wikiKeyword2Pages[key][key2]
    return wikiLines


# Clear out specified unnecessary section from CK12 textbook
#  - Done based on spacing inside text
def clearCK12Section(title, text):
    inSection = False
    spaceCount = 0
    newText = []
    for line in text:
        if inSection == True and spaceCount < 4:
            if line == '': 
                print('EMPTY LINE')
                spaceCount += 1
                continue
            else: 
                spaceCount = 0
                continue
        if inSection == True and spaceCount >= 4:
            inSection == False
        if line == title:
            inSection = True
            spaceCount = 0
            continue
        newText += [line]
    return newText


# Load cleaned paragraphs from CK12 textbook
def getCK12Lines(filename):

    textRaw = open(filename).readlines()
    textRaw = [line.strip('\n') for line in textRaw]
    textRaw = clearCK12Section('References', textRaw)
    textRaw = clearCK12Section('Questions', textRaw)
    textRaw = clearCK12Section('Explore More', textRaw)
    textRaw = clearCK12Section('Review', textRaw)
    textRaw = clearCK12Section('Practice', textRaw)

    cleanText = []
    for line in textRaw:
        if line == '': continue
        if line[-1] == '?': continue
        if len(line) > 8:
            if line[0:7] == "http://": continue
            if line[0:7] == "Figure ": continue
            if line[0:6] == 'Define': continue
            if line[0:5] == 'State': continue
            if line[0:5] == 'Solve': continue
            if line[0:5] == 'Write': continue
            if line[0:8] == 'Describe': continue
            if line[0:3] == 'Use': continue
        if 'Sample Problem' in line or 'Example Problem' in line: continue
        if '_____' in line: continue
        if 'answer the questions below' in line: continue
        if 'Click on the image' in line: continue
        if line[-1] != '.': continue
        cleanText += [line]

    return cleanText


# Given mapping of queries to freebase return values, create next set of queries based on the return values
def getFBSecondOrderQueries(fb1):
    secondOrder = []
    for key in fb1:
        if fb1[key] == None: continue
        for elem in fb1[key]:
            if elem in ['', None]: continue
            if elem in fb1: continue
            secondOrder += [elem]
    return list(set(secondOrder))


# Combine any list of dictionaries together.
#  - Assume keys are strings and values are lists of strings
def combineListofDicts(dictList):
    combinedDict = {}
    for d in dictList:
        for key in d:
            if d[key] in [None, [], '']: continue
            if key.lower() not in combinedDict:
                combinedDict[key.lower()] = []
            for elem in d[key]:
                if elem in ['', None, []]: continue
                combinedDict[key.lower()] += [elem.lower()]
    for key in combinedDict:
        if combinedDict[key] == []: continue
        combinedDict[key] = list(set(combinedDict[key]))
    return combinedDict


# Return the other side of a dictionary (map values to keys)
def makeDict2Sided(d):
    n = {}
    for key in d:
        if d[key] == []: continue
        for elem in d[key]:
            if elem not in n:
                n[elem] = []
            n[elem] += [key]
    for key in n:
        n[key] = list(set(n[key]))
    return n


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
