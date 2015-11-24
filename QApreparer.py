# import countWords as CW
import os, re
import numpy
import editdistance
from spacy.en import English, LOCAL_DATA_DIR
data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
nlp = English(data_dir=data_dir)
questionWordSet = ["who", "what", "when", "where", "how", "why", "which"]

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


# Create one vector per Q-A pair representing basic features regarding format of question and answer:
#  - Number sentences, whether answer is full sentence, fill in the blank types
#  - Correlation between question and answer
#  - Existence of different question words, other words indicating right answer approach
def basicFormatFeatures(pairs):

    features = []
    for pair in pairs:

        currFeatures = []
        question = nlp(pair[2])
        answer = nlp(pair[3])

        # Dummy variable for length of question sentence
        # Distribution of training data sentence lengths is {1: 5574, 2: 3032, 3: 954, 4: 323, 5: 56, 8: 12, 6: 8, 9: 8, 7: 4, 11: 4, 14: 4}
        qLength = sum([1 for sent in question.sents])
        currFeatures += [qLength == 1, qLength == 2, qLength >= 3]

        # Dummy variable for whether answer is full sentence or not
        # In training data 3866 answers are full sentence, 6113 are not
        currFeatures += [answer[-1].text.isalnum() != True and answer[0].text[0].isupper()]

        # Dummy variable for whether question is "fill in the blank"
        # In training data 1117 questions have this feature, 8862 don't
        currFeatures += ["_____" in pair[2]]

        # Variables measuring cosine similarity between question and answer
        # Note distribution is relatively random--poor correlation between similarity and True/False, so buckets also broad
        simQA = question.similarity(answer)
        currFeatures += [simQA, simQA <= 0.5, simQA > 0.5 and simQA <= 0.75, simQA > 0.75]

        # Binary variables measuring existence of question words in last question sentence:
        last = [sent for sent in question.sents][-1].text.lower()
        currFeatures += [x in last for x in questionWordSet]

        # Binary variables measuring existence of words in broader sentence that may be relevant to deciding which approach is best
        # Tried to choose words that occurred with some frequency
        full = " ".join([word.lemma_ for word in question])
        specialWords = ["example", "model", "explain", "describe", "except", "positive", "negative", "cause", "relation"]
        currFeatures += [x in full for x in specialWords]

        features += [currFeatures]

    return numpy.row_stack(tuple(features))


#Concatenate two arrays with the same number of rows
def concat(a,b): return numpy.column_stack((a, b))


# Create one dimensional array of labels (1 if True, 0 if False)
def extractYVector(pairs):
    return numpy.array([int(line[-1]) for line in pairs])


# Extract all relevant nounchunks from the raw question list
def extractNounChunks(rawQA, validationSet = False):
    nounChunks = []
    if validationSet: answerStart = 2
    else: answerStart = 3
    for line in rawQA:
        for chunk in nlp(line[1]).noun_chunks: nounChunks += [chunk.text]
        for x in range(4):
            if "all of the above" in line[answerStart+x].lower() or "none of the above" in line[answerStart+x].lower(): continue
            for chunk in nlp(line[answerStart+x]).noun_chunks: nounChunks += [chunk.text]
    return list(set(nounChunks))


# Convert each QA pair into a single sentence format for search purposes
def convertQAPairToSentence(pair):
    
    question, answer = pair[2], pair[3]
    questionParse = nlp(question)
    
    # Deal with fill in the blank questions with "______"
    blanks = re.findall(r'_+', question)
    if len(blanks) == 1: return question.replace(blanks[0], answer)
    if len(blanks) > 1:
        if ";" in answer: answer = answer.split('; ')
        else: answer = answer.split(', ')
        for i, blank in enumerate(blanks): 
            question = question.replace(blank, answer[i], 1)
        return question
    
    #Deal with fill in the blank questions without "______"
    if question[-1].isalnum() or question[-1] == "-" or question[-1] == ')':
        if question[-2:] == " -": question = question[:-2]
        if question[-1:] == ")": question = question[:-3]
        return question + " " + answer
    
    #Deal with questions asking you to "Choose" or "Identify" the best option
    if "Choose " in question:
        return question.replace('Choose ', '', 1)[:-1] + " is " + answer.lower()
    if "Identify " in question:
        return question.replace('Identify ', '', 1)[:-1] + " is " + answer.lower()
    
    #Deal with full sentence answers:
    if answer[0].isupper() and answer[-1].isalnum() != True:
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
        for sentence in questionParse.sents:
            if "?" in sentence.text:
                for token in sentence:
                    if token.text.lower() in questionWordSet:
                        if token.head.head != token.head: token = token.head
                        start = token.left_edge.idx
                        end = token.right_edge.idx + len(token.right_edge)
                        answer = answer.replace(answer[0], answer[0].upper(), 1)
                        question = question.replace("?", ".")
                        question = question.replace(question[start:end], answer)
                        return question

    # Whenever seems to be special case word, so break it out separately
    if 'whenever' in question.lower():
        return question[:-1] + " " + answer

    # Deal with malformed cases where answer can be placed in question, but question mark is missing
    # Assume the question statement is the last sentence
    for token in [sent for sent in questionParse.sents][-1]:
        if token.text.lower() in questionWordSet:
            if token.head.head != token.head: token = token.head
            start = token.left_edge.idx
            end = token.right_edge.idx + len(token.right_edge)
            answer = answer.replace(answer[0], answer[0].upper(), 1)
            question = question.replace("?", ".")
            question = question.replace(question[start:end], answer)
            return question
    
    # Just return the question and answer joined if none of the above cases are relevant
    # Likely in validation set gibberish
    return " ".join(question, answer)


# Find closest match with wikipedia, report cosine similarity
#  - K-beams implementation: look at k best pages, k best passages, k best sentences
#  - Proximity measured by word2vec cosine similarity, then levenshtein distance
def findClosestWikiMatch(pair, chunk2keys, key2pages, k=5):

    targetSentence = nlp(convertQAPairToSentence(pair))
    # print("Finding closest match for", pair[0])
    
    #Pull out the relevant passages
    chunksRaw = [chunk for chunk in nlp(pair[2]).noun_chunks] + [chunk for chunk in nlp(pair[3]).noun_chunks]
    chunks = [chunk.text for chunk in chunksRaw]
    keys = sum([chunk2keys[chunk] for chunk in chunks if chunk in chunk2keys.keys()],[])
    pages = [key2pages[key] for key in keys if key in key2pages.keys() and key is not None]

    #Focus on the k pages where most/all of the noun chunks appear
    chunksLemmatized = [chunk.lemma_ for chunk in chunksRaw]
    pageScores = []
    for page in pages:
        if page == None: 
            pageScores += [-1]
            continue
        curr = " ".join(sum(page.values(), []))
        pageScores += [sum(chunk in curr for chunk in chunksLemmatized)]
    topKpageScores = sorted(pageScores, reverse=True)[:k]
    topKpages = [pages[pageScores.index(score)] for score in topKpageScores]

    #Choose k best passages matching whole target:
    passages = sum([list(page.values()) for page in topKpages if page is not None], [])
    passages = [nlp(" ".join(passage)) for passage in passages if len(passage) is not 0]    
    passageSimScores = [targetSentence.similarity(passage) for passage in passages]
    topKpassagescores = sorted(passageSimScores, reverse=True)[:k]
    topKpassages = [passages[passageSimScores.index(score)] for score in topKpassagescores]

    #Choose k best sentences matching whole target:
    targetNumSents = len(list(targetSentence.sents))
    sentences = []
    for passage in topKpassages:
        currSentences = list(passage.sents)
        if targetNumSents >= len(currSentences): 
            sentences += [passage]
        for i in range(targetNumSents, len(currSentences)+1):
            rawSentence = [segment.text for segment in currSentences[i-targetNumSents:i] \
                if (segment.text != '') and (segment.text.isspace() != True)]
            if rawSentence == []: continue
            sentences += [nlp(" ".join(rawSentence))]
    sentenceSimScores = [targetSentence.similarity(sentence) for sentence in sentences]
    topKsentencescores = sorted(sentenceSimScores, reverse=True)[:k]
    topKsentences = [sentences[sentenceSimScores.index(score)] for score in topKsentencescores]

    #Check levenshtein distance on lemmatized word basis with target sentence
    targetLemmatized = [tok.lemma_ for tok in targetSentence]
    sentsLemmatized = [[tok.lemma_ for tok in sentence] for sentence in topKsentences]
    lemmatizedDistanceScores = [editdistance.eval(targetLemmatized, currSent) for currSent in sentsLemmatized]
    topKdistanceScores = sorted(lemmatizedDistanceScores)[:k]
    topKLemmatizedSents = [sentsLemmatized[lemmatizedDistanceScores.index(score)] for score in topKdistanceScores]

    print('Results for', pair[0], "--", [float(topKpageScores[0])/len(chunksRaw), topKpassagescores[0], topKsentencescores[0], float(topKdistanceScores[0])/len(targetLemmatized)])
    return [float(topKpageScores[0])/len(chunksRaw), topKpassagescores[0], topKsentencescores[0], float(topKdistanceScores[0])/len(targetLemmatized)]


def getWikiMatchFeatures(pairs, chunk2keys, key2pages, k=5):
    features = [findClosestWikiMatch(pair, chunk2keys, key2pages, k) for pair in pairs]
    return numpy.row_stack(tuple(features))
