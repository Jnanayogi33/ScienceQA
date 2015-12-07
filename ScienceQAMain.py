import QAUtils as utils
import QApreparer as extractor
import learningModels as modeler
import searchText as scraper
import os.path 

##################################################################
print("0. Set global parameters")


#Set number of workers for threaded processes and number of iterations that a pool will run to make sure all work is done
#  - Need fewer threads, more iterations, built-in redundancy in China since API connections unstable (esp. when accessing google/wikipedia)
poolWorkerNum = 20
poolIterations = 3
poolRedundancies = True
cache = '../Dropbox/ScienceQASharedCache/'

##################################################################
print("1. Load all preliminary data, do basic formatting, ")


# Get questions and answers in format ['id', 'question', 'correctAnswer', 'answerA', ..., 'answerD']
#  - In case it is validation set, it will return ['id', 'question', 'answerA', ..., 'answerD']
trainRawQA = extractor.extractQA(cache + 'training_set.tsv')
valRawQA = extractor.extractQA(cache + 'validation_set.tsv', validationSet=True)


# Convert questions and answers into pairs format ['id', 'option' (e.g. 0-3), 'question', 'answer', label (e.g. True/False)]
#  - Q-A pairs where the answer is "all of the above" or "none of the above" were removed
#  - Where "all of the above" or "none of the above" is the right answer, the remaining question-answer pair labels were changed to True or False respectively
#  - In case it is validation set, then just return pairs format ['id', 'option', 'question', 'answer']
trainPairedQA = extractor.convertToQAPairs(trainRawQA)
valPairedQA = extractor.convertToQAPairs(valRawQA, validationSet=True)


# # Extract all noun chunks in the training and validation set
# #  - spacy implementation for deciding what is definition of a 'noun chunk'
# if os.path.isfile(cache + 'nounChunks'): nounChunks = utils.loadData(cache + 'nounChunks')
# else: 
# 	print('Fetching noun chunks')
# 	nounChunks = extractor.extractNounChunksFromPairs(trainPairedQA) + extractor.extractNounChunksFromPairs(valPairedQA)
# 	utils.saveData(nounChunks, cache + 'nounChunks')


# # Download all wikipedia pages matching given set of noun chunks
# #  - Returns two dictionaries: noun chunk --> keywords, and keyword --> page sections --> list of section paragraphs
# #  - Keep separate to minimize memory usage since there would be a lot of redundancy if combined
# #  - Uses pooled worker implementation to speed up downloading
# if os.path.isfile(cache + "WikiChunk2Keywords") and os.path.isfile(cache + "WikiKeyword2Pages"): 
# 	wikiChunk2Keywords = utils.loadData(cache + "WikiChunk2Keywords")
# 	wikiKeyword2Pages = utils.loadData(cache + "WikiKeyword2Pages")
# else:
#     print('Fetching wikipedia')
#     wikiChunk2Keywords, wikiKeyword2Pages = scraper.getWikipediaCompendium(nounChunks, \
#         workerNum = poolWorkerNum, iterations=poolIterations, redundancies=poolRedundancies)
#     utils.saveData(wikiChunk2Keywords, cache + "WikiChunk2Keywords")
#     utils.saveData(wikiKeyword2Pages, cache + "WikiKeyword2Pages")


# # Convert wikipedia data and other textbook data (CK12) into cleaned paragraphs
# #  - Remove lines that are single phrases, questions not useful for extracting logical relationships
# if os.path.isfile(cache + "allTextLines"): utils.allTextLines = utils.loadData(cache + 'allTextLines')
# else:
#     print('Converting wikipedia data')
#     wikiLines = utils.convertWikiPagesToLines(wikiKeyword2Pages)
#     CK12Lines = utils.getCK12Lines(cache + 'Concepts - CK-12 Foundation.txt')
#     utils.allTextLines = [line for line in wikiLines + CK12Lines if len(line.split()) > 5 and '.' in line]
#     utils.saveData(utils.allTextLines, cache + 'allTextLines')
#     del wikiLines, CK12Lines


# # Create index into allTextLines to enable rapid fact checking on limited set of paragraphs
# #  - Leverages SKLearn's Term-frequency inverse document frequency module
# #  - Enables based on given query retrieving most similar passages in allTextLines
# if os.path.isfile(cache + 'allTextIndex.npz') and os.path.isfile(cache + 'allTextVectorizer'):
#     utils.allTextVectorizer = utils.loadData(cache + 'allTextVectorizer')
#     utils.allTextIndex = utils.loadSparseCSR(cache + 'allTextIndex.npz')
#     utils.allTextAnalyzer = utils.allTextVectorizer.build_analyzer()
# else:
#     print('Creating index')
#     utils.allTextVectorizer = utils.TfidfVectorizer(ngram_range=(1,2))
#     utils.allTextIndex = utils.allTextVectorizer.fit_transform(allTextLines)
#     utils.allTextAnalyzer = utils.allTextVectorizer.build_analyzer()
#     utils.saveData(utils.allTextVectorizer, cache + 'allTextVectorizer')
#     utils.saveSparseCSR(utils.allTextIndex, cache + 'allTextIndex.npz')


# # Create a database of likely synonyms based on databases available from freebase and wikipedia
# #  - Used to determine whether words that aren't exact matches are similar
# if os.path.isfile(cache + "synonymCollection"): extractor.synonymCollection = utils.loadData(cache + 'synonymCollection')
# else:
#     print('Collecting synonyms')
#     freebaseSynonymFirstDegree = scraper.getFreebaseSynonyms(nounChunks, workerNum=poolWorkerNum, iterations=poolIterations, redundancies=poolRedundancies)
#     secondOrder = utils.getFBSecondOrderQueries(freebaseSynonymFirstDegree)
#     freebaseSynonymSecondDegree = scraper.getFreebaseSynonyms(secondOrder, workerNum=poolWorkerNum, iterations=poolIterations, redundancies=poolRedundancies)
#     synonymCollection = utils.combineListofDicts([freebaseSynonymFirstDegree, freebaseSynonymSecondDegree, wikiChunk2Keywords])
#     utils.saveData(extractor.synonymCollection, cache + 'synonymCollection')
#     del freebaseSynonymFirstDegree, freebaseSynonymSecondDegree


# # Delete remaining unnecessary files to free up memory
# del wikiKeyword2Pages, wikiChunk2Keywords
# # Loading caches to speed up operations
# if os.path.isfile(cache + "synonymCache"): extractor.synonymCache = utils.loadData(cache + 'synonymCache')
# if os.path.isfile(cache + 'matchingCache'): extractor.matchingCache = utils.loadData(cache + 'matchingCache')
# if os.path.isfile(cache + 'relevantPassageCache'): extractor.relevantPassageCache = utils.loadData(cache + 'relevantPassageCache')

##################################################################
print("2. Create X variables")


# Create one vector per Q-A pair representing basic features regarding format of question and answer:
#  - Number sentences, whether answer is full sentence, fill in the blank types
#  - Existence of different question words, other words indicating right answer approach
#  - spacy word2vec cosine distance between question and answer (own and average of four)
#  - spacy word2vec cosine distance between answer option and other options (own and average of four)
print('- Basic formatting')
trainX = extractor.basicFormatFeatures(trainPairedQA)
valX = extractor.basicFormatFeatures(valPairedQA)
print(trainX.shape)


# Feature measuring proximity of a given Q-A pair to authoritative texts
#  - Q-A combined into a single statement then search carried out to see distance to closest sentence in text
#  - Authoritative text from wikipedia and CK12 free online textbooks for elementary school children
#  - Two measures given--one requiring relatively strict matches, one allowing loose matches
#  - return both absolute value as well as average of other 3 answers
print('- Text match features')
if os.path.isfile(cache + 'trainX'): 
    train_textMatch = utils.loadData(cache + 'trainX')
    print(train_textMatch.shape)
else:
    trainX = extractor.getTextMatchFeatures(trainPairedQA, kList=[100, 10, 100, 1000, 3])
trainX = extractor.concat(trainX, utils.loadData(cache + 'trainX'))

# if os.path.isfile(cache + 'valX'): 
#     valX = extractor.concat(valX, utils.loadData(cache + 'valX'))
# else:
#     valX = extractor.getTextMatchFeatures(valPairedQA, kList=[100, 10, 100, 1000, 3])
# print(trainX.shape)

# Features from the keyword graph from the Aristo paper
#  - size of question graph, size of answer graph, coherence score of answers, coherence score
#    of question keywords, number of pruned words for each Q-A pair
print('- Keyword graph features')
# trainX = extractor.concat(trainX, extractor.getAllMindMapFeatures(trainPairedQA))
# valX = extractor.concat(valX, extractor.getAllMindMapFeatures(valPairedQA))
# print(trainX.shape)

# Features from information retrieval method
# trainX = extractor.concat(trainX, extractor.getIRfeatures(trainPairedQA))
# valX = extractor.concat(valX, extractor.getIRfeatures(valPairedQA))

# Append first order interaction terms of form x_1 * x_2
#  - Note will generate order n^2 interation term features only
print('- Polynomial terms')
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(interaction_only=True)
trainX = poly.fit_transform(trainX)
# valX = poly.fit_transform(valX)
print(trainX.shape)


# Attach array of one vector per Q-A pair representing average of individual word word2vec vectors in both question and answer
#  - new implementation just uses spacy vectors
#  - arrays returned are of the form (m, n) where m is number of data points, n is number of features (300 for word2vec)
# trainX = extractor.concat(trainX, extractor.convertPairsToVectors(trainPairedQA))
# valX = extractor.concat(valX, extractor.convertPairsToVectors(valPairedQA))
# print(trainX.shape)


##################################################################
print("3. Create Y variables and do some sanity checking")


# Create one dimensional array of labels (1 if True, 0 if False)
trainY = extractor.extractYVector(trainPairedQA)


#Saving before we lose anything...
# utils.saveData(trainX, cache + 'trainX')
# utils.saveData(valX, cache + 'valX')
# utils.saveData(trainY, cache + 'trainY')
# utils.saveData(extractor.synonymCache, cache + 'synonymCache')
# utils.saveData(extractor.matchingCache, cache + 'matchingCache')
# utils.saveData(extractor.relevantPassageCache, cache + 'relevantPassageCache')


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

# Vary the C values a bit
candidates["Linear SVC, C = 0.1, feature selection"] = (LinearSVC(class_weight='balanced', C=.1), False, True)
candidates["Linear SVC, C = 0.01, feature selection"] = (LinearSVC(class_weight='balanced', C=.01), False, True)
candidates["Linear SVC, C = 0.001, feature selection"] = (LinearSVC(class_weight='balanced', C=.001), False, True)
candidates["Linear SVC, C = 0.5, feature selection"] = (LinearSVC(class_weight='balanced', C=.5), False, True)



##################################################################
print("5. Take best model, apply to validation set, save results as csv")

# Take the list of model candidates, run 5-fold validation on each, return model settings with best generalization error
#  - 5-fold validation done on actual raw questions, not on Q-A pairs
#  - parameters returned also of the same tuple form as values in the candidates dictionary
#  - if "none of the above" is one of options, will check whether all answers are false
#  - if "all of the above" is one of options, will check whether all answers are true
#  - otherwise will just return answer amongst four with highest confidence score of being true
bestModelParams = modeler.returnBestModel(trainX, trainY, candidates, trainRawQA, trainPairedQA)

# Take best model settings, train on whole training set, then apply trained model to validation set, get answers
#  - Answers in the form "[('id1', 'correctAnswer1"), ...]
validationAnswers = modeler.getValidationAnswers(trainX, trainY, valX, valRawQA, valPairedQA, bestModelParams)

# Print to CSV file in desired submission format for Kaggle
modeler.printAnswersToCSV(validationAnswers, cache + 'submission.csv')