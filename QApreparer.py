# import countWords as CW
import os, re, pickle, copy
import numpy
import util
from Models import WordGraph
import QAUtils as utils
# import editdistance
from spacy.en import English, LOCAL_DATA_DIR
data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
nlp = English(data_dir=data_dir)
questionWordSet = ["who", "what", "when", "where", "how", "why", "which"]

cache = '../Dropbox/ScienceQASharedCache/'

# load mindmaps from local score
if os.path.isfile(cache + 'mindmaps.p'): localMindmaps =  utils.loadData(cache + 'mindmaps.p')
else: localMindmaps = {}

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
    distancesQtoA = {}
    answerVectors = {}
    features = []
    for pair in pairs:
        if pair[0] not in distancesQtoA: distancesQtoA[pair[0]] = []
        if pair[0] not in answerVectors: answerVectors[pair[0]] = []
        currFeatures = []
        question = nlp(pair[2])
        answer = nlp(pair[3])
        answerVectors[pair[0]] += [answer]
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
        distancesQtoA[pair[0]] += [simQA]
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
    distancesAtoA = {}
    for i, pair in enumerate(pairs):
        if pair[0] not in distancesAtoA: 
            distancesAtoA[pair[0]] = []
            count = 0
        currPair = answerVectors[pair[0]][count]
        avgDist = (sum([currPair.similarity(otherPair) for otherPair in answerVectors[pair[0]]]) - 1.0)/(len(answerVectors[pair[0]]) - 1)
        distancesAtoA[pair[0]] += [avgDist]
        features[i] += [avgDist]
        count += 1
    for i, pair in enumerate(pairs):
        features[i] += [sum(distancesQtoA[pair[0]])/len(distancesQtoA[pair[0]])]
        features[i] += [sum(distancesAtoA[pair[0]])/len(distancesAtoA[pair[0]])]
    return numpy.row_stack(tuple(features))


#Concatenate two arrays with the same number of rows
def concat(a,b): return numpy.column_stack((a, b))


# Create one dimensional array of labels (1 if True, 0 if False)
def extractYVector(pairs):
    return numpy.array([int(line[-1]) for line in pairs])


# Normalize special characters known to throw spacy off
def normalizeChars(a):
    a = a.replace('°', ' degrees')
    for elem in re.findall(r' −\w', a):
        a = a.replace(elem, ' negative ' + elem[-1],1)
    for elem in re.findall(r' \+\w', a):
        a = a.replace(elem, ' positive ' + elem[-1],1)
    return a


def kBest(inputs, scoringFunction, k, returnScores=False):
    inputScores = []
    for inputItem in inputs:
        inputScores += [(scoringFunction(inputItem), inputItem)]
    inputScores.sort(key=lambda elem: elem[0], reverse=True)
    if returnScores: return inputScores[:k]
    else: return [elem[1] for elem in inputScores[:k]]


# Convert each QA pair into a single sentence format for search purposes
def convertQAPairToSentence(pair):
    question, answer = normalizeChars(pair[2]), normalizeChars(pair[3])
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
                        # if token.head.head != token.head: token = token.head
                        start = token.left_edge.idx
                        end = token.right_edge.idx + len(token.right_edge)
                        if end-start == len(question):
                            question = question.replace('Which of these ', answer + ' ', 1)
                            question = question.replace('Which of the following ', answer + ' ', 1)
                            question = question.replace('Which of the ', answer + ' ', 1)
                            return question
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
            # if token.head.head != token.head: token = token.head
            start = token.left_edge.idx
            end = token.right_edge.idx + len(token.right_edge)
            answer = answer.replace(answer[0], answer[0].upper(), 1)
            question = question.replace("?", ".")
            question = question.replace(question[start:end], answer)
            return question
    # Just return the question and answer joined if none of the above cases are relevant
    # Likely in validation set gibberish
    return " ".join([question, answer])


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


# Extract all relevant nounchunks from the Q-A pairs list
def extractNounChunksFromPairs(pairs):
    chunkMaster = []
    for pair in pairs:
        targetSentence = nlp(convertQAPairToSentence(pair))
        chunks = [chunk.text for chunk in targetSentence.noun_chunks]
        chunkMaster += chunks
    return list(set(chunkMaster))


def getWordNetSynonyms(word, pos=None):
    if pos is not None and pos[:2] in posTagMap.keys():
        synsets = wordnet.synsets(word, pos=posTagMap[pos[:2]])
    else: synsets = wordnet.synsets(word)
    neighbors = [word]
    for synset in synsets:
        for lemma in synset.lemmas():
            neighbors += [lemma.name().replace('_', ' ')]
    return list(set(neighbors))


def getWordNetAntonyms(word, pos=None):
    if pos is not None and pos[:2] in posTagMap.keys():
        synsets = wordnet.synsets(word, pos=posTagMap[pos[:2]])
    else: synsets = wordnet.synsets(word)
    antonyms = []
    for synset in synsets:
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                antonyms += [antonym.name().replace('_', ' ')]
    return list(set(antonyms))


def getTokenHead(token):
    head = token
    while head != head.head: head = head.head
    return head


def getSentenceRoot(sentence):
    if type(sentence) is str: sentence = nlp(sentence)
    return getTokenHead(sentence[0])


def cleanTokenLists(rawLists):
    cleanLists = []
    for rawList in rawLists:
        curr = []
        for tok in rawList:
            if tok.is_punct or tok.is_space: continue
            if tok.is_stop and tok.lemma_ not in negationWordSet and 'VB' not in tok.tag_: continue
            curr += [tok]
        if curr == []: continue
        cleanLists += [curr]
    return cleanLists


def pullOutClauses(branches, clauseTypes, size):
    changed=True
    while changed:
        changed = False
        for i in range(len(branches)):
            if len(branches[i]) < size: continue
            clauseHeads = [tok for tok in branches[i] if tok.dep_ in clauseTypes]
            if clauseHeads!= []: 
                for clauseHead in clauseHeads:
                    clauseBranch = [tok for tok in list(clauseHead.subtree) if tok in branches[i]]
                    if len(clauseBranch) < size: continue
                    newBranch = [tok for tok in branches[i] if tok not in clauseBranch]
                    if len(newBranch) < size: continue
                    branches[i] = newBranch
                    branches += [clauseBranch]
                    changed = True
                    break
    return branches


def getMainBranches(head, size=5):
    childList = list(head.children)
    branches = []
    headBranch = [head]
    for child in childList:
        if child.tag_ not in ['NN', 'VBN'] and child.dep_ in ['conj', 'advcl', 'adjcl', 'adcl', 'adncl', 'relcl', 'ccomp', 'acl', 'xcomp']: 
            branches += getMainBranches(child, size)
        elif 'comp' in child.dep_ or 'prep' in child.dep_: branches += [list(child.subtree)]
        elif 'RB' in child.tag_ or 'JJ' in child.tag_ or 'IN' in child.tag_: headBranch += list(child.subtree)
        elif 'aux' in child.dep_: headBranch += list(child.subtree)
        elif child.tag_ == 'CC': headBranch += list(child.subtree)
        else: branches += [list(child.subtree)]
    branches = [headBranch] + branches
    if head == head.head: branches = pullOutClauses(branches, ['advcl', 'adjcl', 'adcl', 'adncl', 'relcl', 'ccomp', 'xcomp', 'acl', 'prep', 'appos'], size)
    for i in range(len(branches)):
        branches[i].sort(key=lambda elem: elem.idx)
    branches.sort(key=lambda elem: elem[0].idx)
    return branches


def printBranches(branches):
    for branch in branches: print(' '.join([tok.text for tok in branch]))


def getMainClauses(head, size=10):
    childList = list(head.children)
    if len(childList) == 0: return [[head]]
    clauses = []
    currClause = [head]
    for child in head.children:
        if len(list(child.subtree)) > size:
            currClause += [child]
            clauses += getMainClauses(child, size)
        else: currClause += list(child.subtree)
    if len(currClause) < size and clauses != []:
        minClauseIndex = clauses.index(min(clauses, key=lambda elem: len(elem)))
        clauses[minClauseIndex]  = currClause + clauses[minClauseIndex][1:]
    else: clauses += [currClause]
    return clauses


def matchCost(match):
    if len(match) <= 1: return 0
    prev = match[0]
    cost = 0
    for elem in match[1:]:
        cost -= abs(elem-prev-1)
        prev = elem
    return cost


def kPossibleMatchings(len1, len2, k):
    if (len1, len2, k) in matchingCache: return matchingCache[(len1, len2, k)]
    possible_pairings = [p for p in itertools.permutations(list(range(len2)), len1)]
    topPairings = kBest(possible_pairings, matchCost, k, True)
    matchingCache[(len1, len2, k)] = topPairings
    return topPairings


def getReplacements(key):
    return []    
    # key = " ".join([word for word in key.split() if word.isspace() != True and word not in ['the', 'a', 'an', 'this', 'it'] and word.isalnum()])
    # candidates = []
    # if key in wikiTriples: candidates += wikiTriples[key]
    # if key in textBookTriples: candidates += textBookTriples[key]
    # if key in wikiTriplesAug: candidates += wikiTriplesAug[key]
    # return candidates


def frequencyCalculator(target, candidate):
    candidateLemmas = [tok.lemma_ for tok in candidate]
    targetCopy = [tok for tok in target]
    total = 0.0
    score = 0.0
    for tok in target:
        if tok.lemma_ in candidateLemmas:
            if tok.prob == -10.0 or tok.lemma_ in negationWordSet: score += 40.0
            else: score -= tok.prob
            i = candidateLemmas.index(tok.lemma_)
            candidateLemmas.remove(candidateLemmas[i])
            candidate.remove(candidate[i])
            targetCopy.remove(tok)
        if tok.prob == -10.0 or tok.lemma_ in negationWordSet: total += 40.0
        else: total -= tok.prob
    # for tok in targetCopy:
    #     if candidate == []: break
    #     synProbs = [synonymProb(tok.lemma_, candTok.lemma_, tok.tag_, candTok.tag_) for candTok in candidate]
    #     maxProb = max(synProbs)
    #     maxProbTok = candidate[synProbs.index(maxProb)]
    #     if maxProbTok.prob == -10.0 or maxProbTok in negationWordSet: score += 40.0*maxProb
    #     else: score -= maxProbTok.prob*maxProb 
    #     candidate.remove(maxProbTok)
    for tok in candidate:
        if tok.prob == -10.0 or tok.lemma_ in negationWordSet: total += 40.0
        else: total -= tok.prob
    # print(score/total)
    return score/total


def frequencyBasedSimilarity(target, candidate):
    if type(target) is str: target = nlp(target)
    if type(candidate) is str: candidate = nlp(candidate)
    candidate = [tok for tok in candidate]
    target = [tok for tok in target]
    targetLemmas = [tok.lemma_ for tok in target]
    maxScore = frequencyCalculator(target, candidate)
    for i in range(1,len(target)+1):
        for j in range(len(target)-i+1):
            key = ' '.join(targetLemmas[j:j+i])
            # print('Current key:', key)
            switchOptions = getReplacements(key)
            for option in switchOptions:
                # print('Current option:', option)
                option = [tok for tok in nlp(option)]
                newVersion = target[:j] + option + target[j+i:]
                newScore = frequencyCalculator(newVersion, candidate)
                if newScore > maxScore: maxScore = newScore
    return maxScore


def synonymProb(word1, word2, word1pos=None, word2pos=None):
    if (word1, word2, word1pos, word2pos) in synonymCache:
        return synonymCache[(word1, word2, word1pos, word2pos)]
    if word1pos is not None and word1pos[:2] in posTagMap.keys():
        word1synsets = wordnet.synsets(word1, pos=posTagMap[word1pos[:2]])
    else: word1synsets = wordnet.synsets(word1)
    if word2pos is not None and word2pos[:2] in posTagMap.keys():
        word2synsets = wordnet.synsets(word2, pos=posTagMap[word2pos[:2]])
    else: word2synsets = wordnet.synsets(word2)
    maxMatch = 0.0
    for syn1 in word1synsets:
        for syn2 in word2synsets:
            currMatch = syn1.path_similarity(syn2)
            if currMatch == None: continue
            if currMatch > maxMatch: maxMatch = currMatch
    synonymCache[(word1, word2, word1pos, word2pos)] = maxMatch
    return maxMatch


def strictSentenceMatchScore(target, candidate, k):
    # Derive basic needed data structures where needed
    if type(target) is str: target = nlp(target)
    if type(candidate) is str: candidate = nlp(candidate)
    targetHead = getSentenceRoot(target)
    candidateHead = getSentenceRoot(candidate)
    targetBranches = cleanTokenLists(getMainBranches(targetHead))
    candidateBranches = cleanTokenLists(getMainBranches(candidateHead))
    if len(targetBranches) == 0 or len(candidateBranches) == 0: return 0.0
    if len(targetBranches) + len(candidateBranches) > 20: return 0.0
    if len(targetBranches) > len(candidateBranches): return 0.0
    # Switch out synonyms
    for targetBranch in targetBranches:
        for targetTok in targetBranch:
            for i, candidateBranch in enumerate(candidateBranches):
                for j, candidateTok in enumerate(candidateBranch):
                    if synonymProb(candidateTok.lemma_, targetTok.lemma_, candidateTok.tag_, targetTok.tag_) == 1.0:
                        candidateBranches[i][j] = nlp(targetTok.text)[0]
    # for branch in targetBranches: print(" ".join([tok.text for tok in branch]))
    # for branch in candidateBranches: print(" ".join([tok.text for tok in branch]))
    # Get best possible score for given sentence
    maxScore = 0.0
    matches = kPossibleMatchings(len(targetBranches), len(candidateBranches), k)
    if matches[-1][0] == 0: weight = 0.0
    else: weight = math.log(0.5)/matches[-1][0]
    for match in matches:
        currScore = 1.0
        for i,j in enumerate(match[1]):
            tempScore = frequencyBasedSimilarity(targetBranches[i],candidateBranches[j])
            currScore *= tempScore
        currScore *= math.exp(weight*match[0])
        if currScore >= maxScore: 
            maxScore = currScore
    return math.pow(maxScore, 1/len(targetBranches))


def getSubsentences(sentence):
    if type(sentence) is str: sentence = nlp(sentence)
    if sentence.text == '' or sentence.text.isspace(): return []
    subSentencesRaw = [nlp(subSentence.text) for subSentence in list(sentence.sents)]
    subSentences = []
    for subSentence in subSentencesRaw:
        if len(subSentence) == 0: continue
        frontHead = getTokenHead(subSentence[0])
        backHead = getTokenHead(subSentence[-1])
        if frontHead != backHead: 
            frontSent = " ".join([child.text for child in frontHead.subtree])
            backSent = " ".join([child.text for child in backHead.subtree])
            subSentences += [nlp(frontSent), nlp(backSent)]
        else: subSentences += [subSentence]
    subSentFinal = []
    for subSentence in subSentences:
        if len(subSentence) == 0: continue
        head = getTokenHead(subSentence[0])
        conjunctions = [child for child in head.children if child.dep_ == 'conj']
        if [child for child in head.children if child.dep_ == 'nsubj'] == [] or \
            [child for child in head.children if child.dep_ == 'dobj'] == [] or \
            conjunctions == []: 
            subSentFinal += [subSentence]
            continue
        removalIndices = []
        for conjunction in conjunctions:
            if [child for child in conjunction.children if child.dep_ == 'nsubj'] == [] or \
                [child for child in conjunction.children if child.dep_ == 'dobj'] == []: 
                continue
            conjSent = " ".join([child.text for child in conjunction.subtree])
            subSentFinal += [nlp(conjSent)]
            removalIndices += list(range(conjunction.left_edge.idx, conjunction.right_edge.idx + len(conjunction.right_edge) + 1))
        subSentFinal += [nlp(''.join([a for i,a in enumerate(subSentence.text) if i not in removalIndices]))]
    subSentFinal = [subSent for subSent in subSentFinal if len(subSent) > 0 and subSent[0].is_space != True]
    return subSentFinal


def findClosestTextMatch(pair, kList=[100, 10, 100, 1000, 3]):
    #Get data into standard format
    if type(pair) is list: targetSentence = nlp(convertQAPairToSentence(pair))
    elif type(pair) is str: targetSentence = nlp(pair)
    else: targetSentence = pair 
    targetSubsentences = getSubsentences(targetSentence)
    print(pair)
    # Choose top relevant paragraphs
    if (targetSentence.text, kList[0]) in relevantPassageCache:
        topParagraphsRough = relevantPassageCache[(targetSentence.text, kList[0])]
    else: 
        topParagraphsRough = utils.getRelevantPassages(targetSentence.text, kList[0])
        relevantPassageCache[(targetSentence.text, kList[0])] = topParagraphsRough
    topParagraphsRough = [nlp(normalizeChars(p)) for p in topParagraphsRough]
    topParagraphs = kBest(topParagraphsRough, lambda paragraph: targetSentence.similarity(paragraph), kList[1])
    # Choose top relevant sentences then get final score
    finalScoreRaw = 1.0
    finalScore = 1.0
    sentences = sum([getSubsentences(paragraph) for paragraph in topParagraphs], [])
    for subsent in targetSubsentences:
        topSentencesRaw = kBest(sentences, lambda sentence: subsent.similarity(sentence), kList[2], returnScores=True)
        topSentences = kBest(topSentencesRaw, lambda sentence: strictSentenceMatchScore(subsent, sentence[1], kList[3]), kList[4], returnScores=True)
        # for sentence in topSentences: print(sentence[0], sentence[1][1].text)
        finalScoreRaw *= topSentencesRaw[0][0]
        finalScore *= topSentences[0][0]
    return math.pow(finalScore, 1/len(targetSubsentences)), math.pow(finalScoreRaw, 1/len(targetSubsentences))


def getQuestionAverages(qids, values):
    scores = {}
    for i, qid in enumerate(qids):
        if qid not in scores: scores[qid] = []
        scores[qid] += [values[i]]
    return [sum(scores[qid])/len(scores[qid]) for qid in qids]


def getTextMatchFeatures(pairs, kList=[100, 10, 100, 1000, 3]):
    features = [findClosestTextMatch(pair, kList) for pair in pairs]
    avg0 = getQuestionAverages([p[0] for p in pairs], [f[0] for f in features])
    avg1 = getQuestionAverages([p[0] for p in pairs], [f[1] for f in features])
    features = [[features[i][0], features[i][1], avg0[i], avg1[i]] for i in range(len(features))]
    return numpy.row_stack(tuple(features))

######################################################
# Keyword Graph Feature Extractor
######################################################
def normalizeFeatureVector(features):
    newFeatures = list(features)

    # get row sums
    numFeatures = len(newFeatures[0])
    rowSums = [0] * (numFeatures)
    for row in newFeatures:
        for i in range(numFeatures):
            rowSums[i] += row[i]
    
    # normalize features
    for row in newFeatures:
        for i in range(numFeatures):
            row[i] = row[i]/rowSums[i]

    return newFeatures

def mindmapFeatureExtractor(pair):
    N = 6
    features = []
    questionText, answerText = pair[2], pair[3]

    # get mindmap ==> wordGraph
    if questionText in localMindmaps:
        # print('Mindmap accessed from local store!')
        wordGraph = localMindmaps[questionText]
    else:
        # save mindmap to localMindmaps
        wordGraph = WordGraph(questionText, N)
        localMindmaps[questionText] = wordGraph
        utils.saveData(localMindmaps, cache + 'mindmaps.p')
        print('Mindmaps saved.')

    # make a deep copy of wordGraph to prevent cross-answer
    # contamination ==> save as questionGraph
    questionGraph = copy.deepcopy(wordGraph)

    sizeOfQuestionGraph = len(questionGraph.graph)

    # get coherence score of answer keywords
    answerCoherence = questionGraph.getAnswerScore(answerText)

    # get number of keywords in final graph
    sizeOfAnswerGraph = len(questionGraph.graph)

    # get coherence score of question keywords
    questionCoherence = 0
    for keyword in util.getKeywords(questionText):
        if keyword in questionGraph.graph:
            questionCoherence += wordGraph.coherenceScore(keyword)

    # get number of pruned words
    prunedWords = sizeOfQuestionGraph - sizeOfAnswerGraph

    features.append(sizeOfQuestionGraph)
    features.append(sizeOfAnswerGraph)
    features.append(questionCoherence)
    features.append(answerCoherence)
    features.append(prunedWords)

    return features

def getAllMindMapFeatures(pairs):
    features = []
    for i, pair in enumerate(pairs):
        print('working on {} of {}'.format(i, len(pairs)))
        features.append(mindmapFeatureExtractor(pair))

    features = normalizeFeatureVector(features)
    
    return numpy.row_stack(tuple(features))
