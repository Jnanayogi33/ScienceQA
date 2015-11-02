#Parse inputs into appropriate atomic phrases

import os
import nltk
import copy
from nltk import Tree, ParentedTree
from nltk.parse import stanford
import re
import countWords as CW

os.environ['STANFORD_PARSER'] = './jars'
os.environ['STANFORD_MODELS'] = './jars'
os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk1.8.0_65/bin'
parser = stanford.StanfordParser(model_path="./jars/englishPCFG.ser.gz", java_options='-Xmx2048m')
stanford.StanfordParser()

def getSubTree(sentence, labels):
    positions = sorted(sentence.treepositions(), key=lambda pos: len(pos))
    for position in positions:
        if type(sentence[position]) is unicode: continue
        if sentence[position].label() in labels:
            return position, sentence[position]
    return None, []

def getPrepPhrase(sentence): return getSubTree(sentence, ["PP", "SBAR"])
def getNounPhrase(sentence): return getSubTree(sentence, ["NP"])
def getVerbPhrase(sentence): return getSubTree(sentence, ["VP"])
def getAdverbPhrase(sentence): return getSubTree(sentence, ["ADVP"])

def getSubject(sentence): return getNounPhrase(sentence)
def getPredicate(sentence): return getVerbPhrase(sentence)

def getPrepParse(sentence):
    sentence = sentence.copy(True)
    index, prepPhrase = getPrepPhrase(sentence)
    if index != None and len(index) < 3 and index != ():
        sentence.__delitem__(index)
        if prepPhrase[0].leaves()[0].lower() in [u'besides', u'beside']:
            return []
        elif len(prepPhrase.leaves())>1 and prepPhrase.leaves()[0:2] in [[u'in', u'addition'], [u'In', u'addition']]:
            return []
        return [prepPhrase[0], prepPhrase, sentence]
    return []

def getSVOParse(sentence):
    subjIndex, subjPhrase = getNounPhrase(sentence)
    if subjIndex == None: return []
    verbIndex, verbPhrase = getVerbPhrase(sentence)
    if verbIndex == None: return []
    objIndex, objPhrase = getNounPhrase(sentence[verbIndex])
    if objIndex == None: return []
    return [verbPhrase[0], subjPhrase, objPhrase]

def getSVBroadParse(sentence):
    subjIndex, subjPhrase = getSubject(sentence)
    if subjIndex == None: return []
    verbIndex, verbPhrase = getPredicate(sentence)
    if verbIndex == None: return []
    return [verbPhrase[0], subjPhrase, verbPhrase]

def getPrunedVariations(parse):
    if len(parse) != 3: return []
    setParses = [parse]

    if len(parse[1].leaves()) > 10:

        index1, advPhrase1 = getAdverbPhrase(parse[1])
        if index1 != None:
            parseCopy = parse[1].copy(True)
            parseCopy.__delitem__(index1)
            setParses += [[parse[0], parseCopy, parse[2]]]

        index3, prepPhrase = getPrepPhrase(parse[1])
        if index3 != None:
            parseCopy = parse[1].copy(True)
            parseCopy.__delitem__(index3)
            setParses += [[parse[0], parseCopy, parse[2]]]

    if len(parse[2].leaves()) > 10:

        index2, advPhrase2 = getAdverbPhrase(parse[2])
        if index2 != None:
            parseCopy = parse[2].copy(True)
            parseCopy.__delitem__(index2)
            setParses += [[parse[0], parse[1], parseCopy]]

        index4, prepPhrase = getPrepPhrase(parse[2])
        if index4 != None:
            parseCopy = parse[2].copy(True)
            parseCopy.__delitem__(index4)
            setParses += [[parse[0], parse[1], parseCopy]]

    return setParses

def getSentParses(sentence):

    if type(sentence) != str or len(sentence.split()) <= 1: return []

    #Convert sentence into Stanford-parsed tree
    sentence = ParentedTree.convert(list(parser.raw_parse(sentence))[0])

    #Split sentences if they contain multiple full sentences separated by ';', etc.
    sentences = []
    if (sentence[0].label() == 'S') and (sentence[0,0].label() == 'S'):
        for i in range(len(sentence[0])):
            sentences += [sentence[0,i]]
    else:
        for i in range(len(sentence)):
            sentences += [sentence[i]]

    #Obtain desired tuple relations
    parsedSents = []
    for sentence in sentences:
        print "Current subsentence", sentence.leaves()
        parsedSents += [getPrepParse(sentence)]
        parsedSents += [getSVBroadParse(sentence)]

    #Basic stupid coreferencing
    defaultSet = False
    for parsedSent in parsedSents:
        if len(parsedSent) == 0: continue
        if parsedSent[1].label() == 'NP' and parsedSent[1][0].label() != 'PRP':
            default = parsedSent[1]
            defaultSet = True
        if parsedSent[1].label() == 'NP' and parsedSent[1][0].label() == 'PRP' and defaultSet:
            parsedSent[1] = default

    return parsedSents


def stripPunct(tokens):
    clean = []
    for token in tokens:
        if token in [u'-LRB-',  u'-RRB-', u'.', u';', u',', u'?', u'!', u':', u';', u'', u"'s", u'``', u'""']: continue
        clean += [token]
    return clean

def getAllParses(compendium, tfidfVals, threshold = 1.0):

    outputParses = []
    for num, key in enumerate(compendium.keys()):
        print "Currently working on topic", num+1, "/", len(compendium.keys())
        topic = [CW.standardizeWords(word) for word in key.split()]
        for subkey in compendium[key].keys():
            subtopic = [CW.standardizeWords(word) for word in subkey.split()]
            for paragraph in compendium[key][subkey]:
                for sentence in nltk.tokenize.sent_tokenize(paragraph):
                    print "Current sentence:", sentence
                    for parse in getSentParses(sentence):
                        if len(parse) == 0: continue
                        rawParse = [stripPunct(parse[i].leaves()) for i in range(3)]
                        for i,part in enumerate(rawParse):
                            rawParse[i] = [CW.standardizeWords(word) for word in part]
                            if i > 0 and CW.getAvgTfIdf(key, tfidfVals, rawParse[i]) < threshold:
                                rawParse[i] += topic
                                rawParse[i] += subtopic
                        printParse(rawParse)
                        outputParses += [rawParse]

    return outputParses

def printParse(parse):
    print parse[0]
    print parse[1]
    print parse[2]

def printParses(parses, range):
    for i in range:
        printParse(parses[i])