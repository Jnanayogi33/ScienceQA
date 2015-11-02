#Convert parses into a graph

import gensim, os
import countWords as CW

class word2vecSimilarity:

    def __init__(self, model):
        self.model = model

    def getCosine(self, x, y):

        xVec = [0]*100
        xCount = 0
        for word in x:
            if not self.model.__contains__(word): continue
            else: xVec += self.model[word]
            xCount += 1

        yVec = [0]*100
        yCount = 0
        for word in y:
            if not self.model.__contains__(word): continue
            else: yVec += self.model[word]
            yCount += 1

        if xCount == 0 or yCount == 0: return False
        xVec = xVec/xCount
        yVec = yVec/yCount
        dotProduct = sum([i*j for (i, j) in zip(xVec, yVec)])

        return dotProduct

    def similar(self, x, y, threshold): return self.getCosine(x,y) > threshold

def trainWord2Vec(path):
    sentences = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            sentences += open(path + file).readlines()
    for i, sentence in enumerate(sentences):
        if i % (len(sentences)/10) == 0: print "Finished standardizing sentence", i, "/", len(sentences)
        sentences[i] = [CW.standardizeWords(word) for word in sentence.split()]
    return gensim.models.Word2Vec(sentences, min_count=1)

class ParseGraph:

    def  __init__(self, parses, similarityModel, targetEdgeNodeRatio=1.5):

        self.parses = parses
        self.similarityModel = similarityModel
        self.question = None
        self.answer = None
        self.targetEdgeNodeRatio = targetEdgeNodeRatio
        self.threshold = 1.0
        self.edgeNum = 0
        self.nodeNum = len(parses)*2
        self.graph = {}

        for i,parse in enumerate(self.parses):
            self.graph[(i,1)] = [(i,2)]
            self.graph[(i,2)] = [(i,1)]
            self.edgeNum+=2

        while self.edgeNum < self.nodeNum*self.targetEdgeNodeRatio:
            self.threshold -= 0.1
            print "Current attempt at threshold level", self.threshold
            for i,parse in enumerate(self.parses):
                if i % (len(self.parses)/10) == 0:
                    print "Finished adding node", i, "/", len(self.parses)
                self.addNode((i,1), self.getNodeValue((i,1)))
                self.addNode((i,2), self.getNodeValue((i,2)))

    def addNode(self, key, value):
        if key not in self.graph.keys(): self.graph[key] = []
        for nodeKey in self.graph.keys():
            if key == nodeKey: continue
            if self.similarityModel.similar(value, self.getNodeValue(nodeKey), self.threshold):
                if key not in self.graph[nodeKey]:
                    self.graph[nodeKey] += [key]
                    self.edgeNum += 1
                if nodeKey not in self.graph[key]:
                    self.graph[key] += [nodeKey]
                    self.edgeNum += 1

    def removeNode(self, key):
        if key not in self.graph.keys(): return
        for nodeKey in self.graph[key]:
            if key in self.graph[nodeKey]:
                self.graph[nodeKey].remove(key)
                self.edgeNum -= 1
        self.edgeNum -= len(self.graph[key])
        self.graph[key] = []

    def addQuestion(self, question, minNewEdges=10):
        origThreshold = self.threshold
        origEdges = self.edgeNum
        self.question = question
        while (self.edgeNum - origEdges) < minNewEdges:
            self.threshold -= 0.1
            self.removeNode('question')
            self.addNode('question', self.question)
        self.threshold = origThreshold

    def addAnswer(self, answer, minNewEdges=10):
        origThreshold = self.threshold
        origEdges = self.edgeNum
        self.answer = answer
        while (self.edgeNum - origEdges) < minNewEdges:
            self.threshold -= 0.1
            self.removeNode('answer')
            self.addNode('answer', self.answer)
        self.threshold = origThreshold

    def resetQA(self):
        self.removeNode('question')
        self.removeNode('answer')

    def getNeighbors(self, key): return self.graph[key]
    def getNodeValue(self, key):
        if key == 'answer': return self.answer
        if key == 'question': return self.question
        return self.parses[key[0]][key[1]]