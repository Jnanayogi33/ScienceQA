import QAUtils as util
import QApreparer as QA
import learningModels as LM


##################################################################
print("1. Load all preliminary data, do basic formatting")

# Get questions and answers in format ['id', 'question', 'correctAnswer', 'answerA', ..., 'answerD']
#  - In case it is validation set, it will return ['id', 'question', 'answerA', ..., 'answerD']
trainRawQA = QA.extractQA('training_set.tsv')
valRawQA = QA.extractQA('validation_set.tsv', validationSet=True)

# Convert questions and answers into pairs format ['id', 'option' (e.g. 0-3), 'question', 'answer', label (e.g. True/False)]
#  - Q-A pairs where the answer is "all of the above" or "none of the above" were removed
#  - Where "all of the above" or "none of the above" is the right answer, the remaining question-answer pair labels were changed to True or False respectively
#  - In case it is validation set, then just return pairs format ['id', 'option', 'question', 'answer']
trainPairedQA = QA.convertToQAPairs(trainRawQA)
valPairedQA = QA.convertToQAPairs(valRawQA, validationSet=True)


##################################################################
print("2. Create X variables")

# Create array of one vector per Q-A pair representing average of individual word word2vec vectors in both question and answer
#  - new implementation just uses spacy vectors
#  - arrays returned are of the form (m, n) where m is number of data points, n is number of features (300 for word2vec)
trainX = QA.convertPairsToVectors(trainPairedQA)
valX = QA.convertPairsToVectors(valPairedQA)

# Create a version of the X array with all entries log-transformed, concatenate it to original X values
#  - if any negative X entries, add |min(X)| + 1 to each entry before log transform, otherwise just add 1 
#  - currently inactive because word2vec vector values are already scaled and in a small range, so this is irrelevant
#  - however leaving this here because in future may be useful specifically if applied for saturated features
# trainX = QA.concat(trainX, QA.createLog(trainX))
# valX = QA.concat(valX, QA.createLog(valX))


## === MOSES - INSERT NEW FEATURES HERE === ##
#  - Create them in form (m,n) where m is number of data points, n is number of features
#  - Use oldX = QA.concat(oldX, newFeatures) to concatenate with existing X variables
#  - Remember to do this on both trainX AND valX because we need to be building up valX for final result


##################################################################
print("3. Create Y variables")

# Create one dimensional array of labels (1 if True, 0 if False)
trainY = QA.extractYVector(trainPairedQA)


##################################################################
print("4. Specify a variety of models we think will be appropriate")

# Store candidates in a dictionary to be passd to a selector in step 5
#  - key is the name of the candidate
#  - value is tuple (model, Bool for whether to do X value standardization, Bool for whether to do feature selection)
#  - X value standardization ensures each feature's mean value is 0, stdev is 1
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

# Try SVM with polynomial kernel order 2 (to capture first-order interactions between features i.e. x_1^2 and x_1x_2)
#  - Currently disabled because generalization error about as good as linear SVC but takes too long to run.
#  - but leaving in because in future with more variables added in form of dummy buckets may become useful
# from sklearn.svm import SVC
# candidates["SVC w/ polynomial kernel (d=2), C = 1.0"] = (SVC(class_weight='balanced', kernel='poly', degree=2), False, False)
# candidates["SVC w/ polynomial kernel (d=2), X normalization, C = 1.0"] = (SVC(class_weight='balanced', kernel='poly', degree=2), True, False)
# candidates["SVC w/ polynomial kernel (d=2), C = 1.0, feature selection"] = (SVC(class_weight='balanced', kernel='poly', degree=2), False, True)
# candidates["SVC w/ polynomial kernel (d=2), X normalization, C = 1.0, feature selection"] = (SVC(class_weight='balanced', kernel='poly', degree=2), True, True)


##################################################################
print("5. Take best model, apply to validation set, save results as csv")

# Take the list of model candidates, run 5-fold validation on each, return model settings with best generalization error
#  - 5-fold validation done on actual raw questions, not on Q-A pairs
#  - parameters returned also of the same tuple form as values in the candidates dictionary
#  - if "none of the above" is one of options, will check whether all answers are false
#  - if "all of the above" is one of options, will check whether all answers are true
#  - otherwise will just return answer amongst four with highest confidence score of being true
bestModelParams = LM.returnBestModel(trainX, trainY, candidates, trainRawQA, trainPairedQA)

# Take best model settings, train on whole training set, then apply trained model to validation set, get answers
#  - Answers in the form "[('id1', 'correctAnswer1"), ...]
validationAnswers = LM.getValidationAnswers(trainX, trainY, valX, valRawQA, valPairedQA, bestModelParams)

# Print to CSV file in desired submission format for Kaggle
LM.printAnswersToCSV(validationAnswers, 'submission.csv')