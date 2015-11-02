import os, gensim, logging
import QAUtils as util
import searchText as ST
import countWords as CW
import phraseParser as PP
import graphBuilder as GB
import graphSolver as GS
import QApreparer as QA

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
os.chdir('C:/Users/Tin Yun/Dropbox/Stanford/CS 221/Final Project/ScienceQA')

print "1. Download compendium based on keywords and save"
keywords = ST.getKeywords('ck12_list_keyword.txt')
util.saveData(keywords, 'WikipediaCK12SearchTerms')
keywords = util.loadData('WikipediaCK12SearchTerms')
compendium = ST.createCompendium(keywords, startIndex=0)
util.saveData(compendium, 'WikipediaCK12Compendium')

print "2. Obtain Tf-Idf values based on the compendium and save"
compendium = util.loadData('WikipediaCK12Compendium')
wordCountByTopic = CW.getWordCountByTopic(compendium)
wordTopicCounts = CW.getWordTopicCounts(wordCountByTopic)
tfidfVals = CW.createTfIdfVals(wordCountByTopic, wordTopicCounts)
util.saveData(tfidfVals, 'WikipediaCK12TfIdfVals')

print "3. Get parses for each sentence in the compendium"
tfidfVals = util.loadData('WikipediaCK12TfIdfVals')
playCompendium = {}
playCompendium['Solar System'] = compendium['Solar System']
parseData = PP.getAllParses(playCompendium, tfidfVals, threshold=2.5)
util.saveData(parseData, 'playListParses')

print "4. Train word2vec model for measuring similarity between nodes"
word2vecTrained = GB.trainWord2Vec('C:/Users/Tin Yun/Dropbox/Stanford/CS 221/Final Project/wikipedia_content_based_on_ck_12_keyword_one_file_per_keyword/')
word2vecTrained.save('baseline_word2vec')
word2vecTrained = gensim.models.Word2Vec.load('baseline_word2vec')
simModel = GB.word2vecSimilarity(word2vecTrained)

print "5. Use the parses and similarity model to create the graph"
parseData = util.loadData('playListParses')
parseGraph = GB.ParseGraph(parseData, simModel, targetEdgeNodeRatio=1.5)
util.saveData(parseGraph, 'playListParseGraph')

print "6. Given input of question and answer options, solve."
print " - Similarity (sim) refers to test based on word2vec cosine similarity"
print " - Jump (jump) refers to test based on shortest path in graph"
parseGraph = util.loadData('playListParseGraph')
QAList = QA.extractQA('solarSystemSet.tsv')

answerIndex = ['A', 'B', 'C', 'D']

simResult = 0
jumpResult = 0
simRightjumpWrong = 0
simWrongjumpRight = 0
bothRight = 0
bothWrong = 0

for QAline in QAList:
    question, answers, answerKey = QA.normalizeQA(QAline)
    simScores = []
    jumpScores = []
    sequences = []
    for answer in answers:
        score, sequence = GS.graphSearchAlgorithm(question, answer, parseGraph, minNewEdges=1)
        jumpScores += [score]
        simScores += [simModel.getCosine(question, answer)]
        sequences += [[sequence]]

    print QAline
    print "Similarity:", answerIndex[simScores.index(max(simScores))]
    print "Jump:", answerIndex[jumpScores.index(min(jumpScores))]
    print sequences

    if answerIndex[simScores.index(max(simScores))] == answerKey:
        simResult += 1
        if answerIndex[jumpScores.index(min(jumpScores))] == answerKey:
            jumpResult += 1
            bothRight += 1
        else:
            simRightjumpWrong += 1
    else:
        if answerIndex[jumpScores.index(min(jumpScores))] == answerKey:
            jumpResult += 1
            simWrongjumpRight += 1
        else:
            bothWrong += 1

print "Total", len(QAList)
print "sim score", simResult
print "jump score", jumpResult
print "sim right jump wrong", simRightjumpWrong
print "sim wrong jump right", simWrongjumpRight
print "both right", bothRight
print "both wrong", bothWrong