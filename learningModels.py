import numpy as np
import collections, csv
from sklearn import preprocessing
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Run linear kernel SVC and l1 penalty on X and Y, return index of features with non-zero coefficients
#  - used for feature selection
#  - l1 penalty norm pushes individual weight values to 0 if they provide little/no information
def getNonZeroCoefficientFeatures(X, Y, choiceC=1.0):
    model = LinearSVC(C=choiceC, class_weight='balanced', penalty='l1', dual=False)
    model.fit(X,Y)
    return [i for i in range(len(model.coef_.T)) if model.coef_.T[i] != 0.0]


# Scale X values to have mean 0 std 1
def preprocess(X): return preprocessing.scale(X)


# Create mapping from the predicted answers made by the model and the scores to the pair rows
#  - returns mapping with key = ('id', 'answerOption') and value = (prediction int, score probability)
def createPredictionMap(pairs, model, X):
    predictions = model.predict(X)
    probabilities = model.decision_function(X) 
    mapping = {}
    for i, line in enumerate(pairs):
        mapping[(line[0], line[1])] = (int(predictions[i]), probabilities[i])
    return mapping


# go from T-F answers to actual A-D answers
#  - Note this function is relatively complex in order to deal with variety of validation set use cases
def getFinalAnswers(answerMapping, rawQuestions, validationSet=False):
    
    answerOptions = ['A', 'B', 'C', 'D']
    answers = []
    if validationSet: answerStart = 2
    else: answerStart = 3
    
    for question in rawQuestions:
        qID = question[0]
        options = list(range(4))
        lowerCaseAnswers = [a.lower() for a in question[answerStart:]]
        assert(len(options) == len(lowerCaseAnswers))
        
        noneExist = None
        allExist = None

        if "none of the above" in " ".join(lowerCaseAnswers):
            noneExist = [i for i, ans in enumerate(lowerCaseAnswers) if "none of the above" in ans]
        if "all of the above" in " ".join(lowerCaseAnswers): 
            allExist = [i for i, ans in enumerate(lowerCaseAnswers) if "all of the above" in ans]

        if noneExist != None: 
            for i in noneExist: options.remove(i)
        if allExist != None: 
            for i in allExist: options.remove(i)

        if noneExist != None:
            if sum([(answerMapping[(qID, i)][0] == 0) for i in options]) == len(options): 
                answers += [answerOptions[noneExist[0]]]
                continue
        if allExist != None:
            if sum([(answerMapping[(qID, i)][0] == 1) for i in options]) == len(options):
                answers += [answerOptions[allExist[0]]]
                continue

        answerProbs = [answerMapping[(qID, i)][1] for i in options]
        answers += [answerOptions[np.argmax(answerProbs)]]
    
    return answers


# Calculate percent of answers correct
#  - answers is list of letters e.g. 'A', 'B'
#  - questions is the raw QA questions training set
def calcFinalAccuracy(answers, questions):
    correct = 0.0
    for i, line in enumerate(questions):
        if answers[i] == line[2]: correct += 1.0
    return correct/float(len(answers))


# Run nfold validation for given model, return training and test error
#  - testing actually done on raw questions (A-D format), not on pair questions (T-F format)
#  - shuffling done by index of question id to randomize
def multipleChoiceNFoldsValidation(X, Y, model, QA_raw, QA_paired, folds, featureSelection):
    
    index = [q[0] for q in QA_raw]
    shuffle(index)
    index = np.array(index)
    trainScores = []
    testScores = []

    for i in range(folds):

        lowSplit = int(i*len(QA_raw)/folds)
        highSplit = int((i+1)*len(QA_raw)/folds)
        
        trainRawIndex = index[list(range(0,lowSplit)) + list(range(highSplit,len(QA_raw)))]
        testRawIndex = index[lowSplit:highSplit]
        
        trainRaw = [line for line in QA_raw if line[0] in trainRawIndex]
        testRaw = [line for line in QA_raw if line[0] in testRawIndex]
        
        trainPairs = [pair for pair in QA_paired if pair[0] in trainRawIndex]
        testPairs = [pair for pair in QA_paired if pair[0] in testRawIndex]
        
        trainPairedIndex = [i for i in range(len(QA_paired)) if QA_paired[i][0] in trainRawIndex]
        testPairedIndex = [i for i in range(len(QA_paired)) if QA_paired[i][0] in testRawIndex]

        trainX = X[trainPairedIndex]
        trainY = Y[trainPairedIndex]
        testX = X[testPairedIndex]
        testY = Y[testPairedIndex]

        if featureSelection:
            features = getNonZeroCoefficientFeatures(trainX,trainY)
            trainX = trainX[:,features]
            testX = testX[:,features]

        model.fit(trainX, trainY)

        trainPredictionMap = createPredictionMap(trainPairs, model, trainX)
        testPredictionMap = createPredictionMap(testPairs, model, testX)
        trainFinalAnswers = getFinalAnswers(trainPredictionMap, trainRaw)
        testFinalAnswers = getFinalAnswers(testPredictionMap, testRaw)
        trainScores += [calcFinalAccuracy(trainFinalAnswers, trainRaw)]
        testScores += [calcFinalAccuracy(testFinalAnswers, testRaw)]

    return sum(trainScores)/len(trainScores), sum(testScores)/len(testScores)


# Take the list of model candidates, run 5-fold validation on each, return model settings with best generalization error
#  - 5-fold validation done on actual raw questions, not on Q-A pairs
#  - parameters returned also of the same tuple form as values in the candidates dictionary
#  - if "none of the above" is one of options, will check whether all answers are false
#  - if "all of the above" is one of options, will check whether all answers are true
#  - otherwise will just return answer amongst four with highest confidence score of being true
def returnBestModel(X, Y, candidates, QA_raw, QA_paired, folds=5):
    currMax = 0
    currBest = None
    for key in candidates.keys():
        currX = X
        if candidates[key][1] == True: currX = preprocess(currX)
        trainScore, testScore = multipleChoiceNFoldsValidation(currX,Y,candidates[key][0],QA_raw, QA_paired, folds, candidates[key][2])
        print(key, "scores -- Train:", trainScore, "Test:", testScore)
        if testScore > currMax:
            currMax = testScore
            currBest = candidates[key]
    return currBest


# Take best model settings, train on whole training set, then apply trained model to validation set, get answers
#  - Answers in the form "[('id1', 'correctAnswer1"), ...]
def getValidationAnswers(trainX, trainY, valX, valRawQA, valPairedQA, modelParams):
    
    if modelParams[1] == True: 
        trainX = preprocess(trainX)
        valX = preprocess(valX)
    
    if modelParams[2] == True: 
        features = getNonZeroCoefficientFeatures(trainX,trainY)
        trainX[:,features]
        valX[:,features]

    model = modelParams[0]
    model.fit(trainX,trainY)
    valPredictionMap = createPredictionMap(valPairedQA, model, valX)
    valFinalAnswers = getFinalAnswers(valPredictionMap, valRawQA, validationSet=True)

    return [(valRawQA[i][0], valFinalAnswers[i]) for i in range(len(valFinalAnswers))]


# Print to CSV file in desired submission format for Kaggle
def printAnswersToCSV(answers, filename):
    with open(filename, 'w') as csvfile:
        submission = csv.writer(csvfile, delimiter=",")
        submission.writerow(['id', 'correctAnswer'])
        for answer in answers:
            submission.writerow([answer[0],answer[1]])