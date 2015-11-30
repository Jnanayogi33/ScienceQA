import QAUtils as utils
import QApreparer as extractor
import learningModels as modeler
import searchText as scraper

##################################################################
print("0. Set global parameters")

#Set number of workers for threaded processes and number of iterations that a pool will run to make sure all work is done
#  - Need fewer threads, more iterations, built-in redundancy in China since API connections unstable and guarded
poolWorkerNum = 30
poolIterations = 2
poolRedundancies = True


##################################################################
print("1. Load all preliminary data, do basic formatting, ")

# Get questions and answers in format ['id', 'question', 'correctAnswer', 'answerA', ..., 'answerD']
#  - In case it is validation set, it will return ['id', 'question', 'answerA', ..., 'answerD']
trainRawQA = extractor.extractQA('ScienceQASharedCache/training_set.tsv')
# valRawQA = extractor.extractQA('ScienceQASharedCache/validation_set.tsv', validationSet=True)

# Convert questions and answers into pairs format ['id', 'option' (e.g. 0-3), 'question', 'answer', label (e.g. True/False)]
#  - Q-A pairs where the answer is "all of the above" or "none of the above" were removed
#  - Where "all of the above" or "none of the above" is the right answer, the remaining question-answer pair labels were changed to True or False respectively
#  - In case it is validation set, then just return pairs format ['id', 'option', 'question', 'answer']
trainPairedQA = extractor.convertToQAPairs(trainRawQA)
# valPairedQA = extractor.convertToQAPairs(valRawQA, validationSet=True)

# Extract all noun chunks in the training and validation set
#  - implementation uses pool of worker threads to speed up downloading
#  - spacy implementation for deciding noun chunks
# trainNounChunks = extractor.extractNounChunks(trainRawQA)
# valNounChunks = extractor.extractNounChunks(valRawQA, validationSet=True)

# Download all wikipedia pages matching given set of noun chunks
#  - Returns two dictionaries: noun chunk --> keywords, and keyword --> page sections --> list of section paragraphs
#  - Keep separate to minimize memory usage since there would be a lot of redundancy if combined
# wikiChunk2Keywords, wikiKeyword2Pages = scraper.getWikipediaCompendium(trainNounChunks + valNounChunks, \
#     workerNum = poolWorkerNum, iterations=poolIterations, redundancies=poolRedundancies)
# wikiChunk2Keywords = utils.loadData("ScienceQASharedCache/WikiChunk2Keywords")
# wikiKeyword2Pages = utils.loadData("ScienceQASharedCache/WikiKeyword2Pages")

# Download all freebase triples given list of noun chunks
#  - Returns 2 dictionaries: chunk --> list of mids, and mid --> list of triples
#  - Triples in format [[name, "Has property " + property, value['text']], mid of third element in triple]
# freebaseChunk2Mids, freebaseMid2Triples = scraper.getFreebaseCompendium(trainNounChunks[:500], \
    # workerNum = poolWorkerNum, iterations=poolIterations, redundancies=poolRedundancies)
# utils.saveData(freebaseChunk2Mids, 'ScienceQASharedCache/FreebaseChunk2Mids')
# utils.saveData(freebaseMid2Triples, 'ScienceQASharedCache/FreebaseMid2Triples')


##################################################################
print("2. Create X variables")

# Create one vector per Q-A pair representing basic features regarding format of question and answer:
#  - Number sentences, whether answer is full sentence, fill in the blank types
#  - Correlation between question and answer
#  - Existence of different question words, other words indicating right answer approach
trainX = extractor.basicFormatFeatures(trainPairedQA)
# valX = extractor.basicFormatFeatures(valPairedQA)
print('- Basic formatting complete')

# k-beam search top wikipedia text match features (UNDER DEVELOPMENT)
# trainX = extractor.concat(trainX, extractor.getWikiMatchFeatures(trainPairedQA, wikiChunk2Keywords, wikiKeyword2Pages, k=5))
# valX = extractor.concat(valX, extractor.getWikiMatchFeatures(valPairedQA, wikiChunk2Keywords, wikiKeyword2Pages, k=5))


## === MOSES - INSERT NEW FEATURES HERE === ##
#  - Create them in form (m,n) where m is number of data points, n is number of features
#  - Use oldX = extractor.concat(oldX, newFeatures) to concatenate with existing X variables
#  - Remember to do this on both trainX AND valX because we need to be building up valX for final result
# trainX = extractor.concat(trainX, extractor.getAllMindMapFeatures(trainPairedQA))

# Append first order interaction terms of form x_1 * x_2
#  - Note will generate order n^2 features, so do this only for most critical features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(interaction_only=True)
trainX = poly.fit_transform(trainX)
# valX = poly.fit_transform(valX)

# Attach array of one vector per Q-A pair representing average of individual word word2vec vectors in both question and answer
#  - new implementation just uses spacy vectors
#  - arrays returned are of the form (m, n) where m is number of data points, n is number of features (300 for word2vec)
trainX = extractor.concat(trainX, extractor.convertPairsToVectors(trainPairedQA))
# valX = extractor.concat(valX, extractor.convertPairsToVectors(valPairedQA))

##################################################################
print("3. Create Y variables")

# Create one dimensional array of labels (1 if True, 0 if False)
trainY = extractor.extractYVector(trainPairedQA)

#Saving before we lose anything...
utils.saveData(trainX, 'ScienceQASharedCache/trainX')
# utils.saveData(trainX, 'ScienceQASharedCache/valX')
# utils.saveData(trainY, 'ScienceQASharedCache/trainY')

##################################################################
print("4. Specify a variety of models we think will be appropriate")

# Store candidates in a dictionary to be passd to a selector in step 5
#  - key is the name of the candidate
#  - value is tuple (model, Bool for whether to do X value standardization, Bool for whether to do feature selection)
#  - X value standardization ensures each feature's mean value is 0, stdev is 1 (SVC is not linear invariant)
#  - Feature selection uses linearSVM + l1 norm on one pass of data, removes features with 0 coefficients
candidates = {}

# Use logistic regression because most intuitive model for 0-1 classification selection, no assumptions on prior feature data shape
from sklearn.linear_model import LogisticRegression
candidates["Logistic regression, C = 1.0"] = (LogisticRegression(class_weight='balanced'), False, False)
candidates["Logistic regression, X normalization, C = 1.0"] = (LogisticRegression(class_weight='balanced'), True, False)
candidates["Logistic regression, C = 1.0, feature selection"] = (LogisticRegression(class_weight='balanced'), False, True)
candidates["Logistic regression, X normalization, C = 1.0, feature selection"] = (LogisticRegression(class_weight='balanced'), True, True)

# Try SVM with linear kernel (performance should be similar to logit regression)
from sklearn.svm import LinearSVC
candidates["Linear SVC, C = 1.0"] = (LinearSVC(class_weight='balanced'), False, False)
candidates["Linear SVC, X normalization, C = 1.0"] = (LinearSVC(class_weight='balanced'), True, False)
candidates["Linear SVC, C = 1.0, feature selection"] = (LinearSVC(class_weight='balanced'), False, True)
candidates["Linear SVC, X normalization, C = 1.0, feature selection"] = (LinearSVC(class_weight='balanced'), True, True)


##################################################################
print("5. Take best model, apply to validation set, save results as csv")

# Take the list of model candidates, run 5-fold validation on each, return model settings with best generalization error
#  - 5-fold validation done on actual raw questions, not on Q-A pairs
#  - parameters returned also of the same tuple form as values in the candidates dictionary
#  - if "none of the above" is one of options, will check whether all answers are false
#  - if "all of the above" is one of options, will check whether all answers are true
#  - otherwise will just return answer amongst four with highest confidence score of being true
bestModelParams = modeler.returnBestModel(trainX, trainY, candidates, trainRawQA, trainPairedQA)

import pdb; pdb.set_trace()  # breakpoint 593f617b //

# Take best model settings, train on whole training set, then apply trained model to validation set, get answers
#  - Answers in the form "[('id1', 'correctAnswer1"), ...]
validationAnswers = modeler.getValidationAnswers(trainX, trainY, valX, valRawQA, valPairedQA, bestModelParams)

# Print to CSV file in desired submission format for Kaggle
modeler.printAnswersToCSV(validationAnswers, 'ScienceQASharedCache/submission.csv')