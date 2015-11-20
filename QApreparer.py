# import countWords as CW
import os
import numpy
from spacy.en import English, LOCAL_DATA_DIR
data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
nlp = English(data_dir=data_dir)


# Get questions and answers in format ['id', 'question', 'correctAnswer', 'answerA', ..., 'answerD']
#  - In case it is validation set, it will return ['id', 'question', 'answerA', ..., 'answerD']
def extractQA(QAfile, validationSet=False):
    sentences = open(QAfile).readlines()
    result = [sentence.strip("\n").split("\t") for sentence in sentences[1:]]
    for line in result:
        if validationSet: assert(len(line) == 6)
        else: assert(len(line) == 7)
    return result


# Convert questions and answers into pairs format ['id', 'option' (e.g. 0-3), 'question', 'answer', label (e.g. True/False)]
#  - Q-A pairs where the answer is "all of the above" or "none of the above" were removed
#  - Where "all of the above" or "none of the above" is the right answer, the remaining question-answer pair labels were changed to True or False respectively
#  - In case it is validation set, then just return pairs format ['id', 'option', 'question', 'answer']
def convertToQAPairs(raw, validationSet=False):
    QAPairs = []
    answers = ['A', 'B', 'C', 'D']
    if validationSet:
        for line in raw:
            for x in range(len(answers)):
                if "all of the above" in line[2+x].lower() or "none of the above" in line[2+x].lower(): continue
                QAPairs += [[line[0], x, line[1], line[2+x]]]
    else:
        for line in raw:
            if "all of the above" in line[6].lower() and line[2] == 'D':
                for x in range(len(answers) - 1): QAPairs += [[line[0], x, line[1], line[3+x], True]]
            elif "none of the above" in line[6].lower() and line[2] == 'D':
                for x in range(len(answers) - 1): QAPairs += [[line[0], x, line[1], line[3+x], False]]
            else:
                for x in range(len(answers)): 
                    if "all of the above" in line[3+x].lower() or "none of the above" in line[3+x].lower(): continue
                    QAPairs += [[line[0], x, line[1], line[3+x], (line[2] == answers[x])]]
    return QAPairs


# Create one vector per Q-A pair representing average of individual word word2vec vectors in both question and answer
#  - new implementation just uses spacy vectors
def convertPairsToVectors(pairs):
    return numpy.row_stack(tuple([nlp(" ".join([line[2],line[3]])).vector for line in pairs]))


# Create a version of the X array with all entries log-transformed, concatenate it to original X values
#  - if any negative X entries, add |min(X)| + 1 to each entry before log transform, otherwise just add 1 
def createLog(X): 
    if numpy.min(X) < 0: a = -numpy.min(X) + 1.0
    else: a = 1.0 
    X = X + a
    return numpy.log(X)


#Concatenate two arrays with the same number of rows
def concat(a,b): return numpy.column_stack((a, b))


# Create one dimensional array of labels (1 if True, 0 if False)
def extractYVector(pairs):
    return numpy.array([int(line[-1]) for line in pairs])


# Convert each QA pair into a single sentence format for search purposes
def convertQAPairToSentence(question, answer):

    # Deal with statements of the form "_____"
    # Note in training set shortest blank is 9 underscores and longest is 16, only with one blank or two blanks
    answer = answer.strip("\n")
    numBlanks = question.count("_________")
    if numBlanks == 1:
        match = re.search(r'_+', question)
        return question.replace(match.group(0), answer)
    if numBlanks == 2:
        if "," in answer: answer = answer.split(", ")
        else: answer = answer.split("; ")  
        match = re.search(r'_+', question)
        question = question.replace(match.group(0), answer[0], 1)
        match = re.search(r'_+', question)
        return question.replace(match.group(0), answer[1])

    #Deal with statements of completion without "______"
    if question[-1].isalpha() or question[-1].isnumeric():
        if question[-2:] == " -": question = question[:-2]
        return question + " " + answer
    questionParse = nlp(question)

    #Deal with full sentence answers:
    if answer[0].isupper() and answer[-1] == ".":
        answer = answer.replace(answer[0], answer[0].lower(), 1)
        answer = "That " + answer[:-1]
        split = None
        for sentence in questionParse.sents:
            if "?" in sentence.text: split = sentence[0].idx
        if split == None: return "ERROR on question: " + question
        qPart1 = question[:split]
        qPart2 = question[split:]
        qPart2 = qPart2.replace(qPart2[0], qPart2[0].lower(), 1)
        qPart2 = qPart2.replace("?", ".", 1)
        return qPart1 + answer + " is " + qPart2

    #Deal with multi-sentence question statements where answer can be placed in question:
    if "?" in question:
        questionParse = nlp(question)
        for sentence in questionParse.sents:
            if "?" in sentence.text:
                for token in sentence:
                    if token.text.lower() in questionWordSet:
                        while token.head.head != token.head: token = token.head
                        start = token.left_edge.idx
                        end = token.right_edge.idx + len(token.right_edge)
                        answer = answer.replace(answer[0], answer[0].upper(), 1)
                        question = question.replace("?", ".")
                        question = question.replace(question[start:end], answer)
                        return question
    else: return "ERROR on question: " + question