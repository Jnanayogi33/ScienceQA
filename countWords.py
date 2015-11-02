#Create word2vec and tf-idf vectors

import collections, re, math
import sklearn, nltk

porterStem = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def getWordnetPOS(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return 'n'

def standardizeWords(word):
    word = word.strip('\n')
    word = re.sub(r'[^\w\s\.\'\-]','',word)
    word = re.sub(r'\. ','',word + ' ')
    word = re.sub(r' ','',word)
    tag = getWordnetPOS(nltk.pos_tag([word])[0][1])
    word = lemmatizer.lemmatize(word, tag)
    word = porterStem.stem(word)
    return word.lower()

def getWordCountByTopic(compendium):
    wordCountByTopic = {}
    for i,key in enumerate(compendium.keys()):
        print "Working on topic", i, "/", len(compendium.keys())
        wordCountByTopic[key] = collections.Counter()
        for subkey in compendium[key].keys():
            for line in compendium[key][subkey]:
                for word in line.split():
                    word = standardizeWords(word)
                    if word != None:
                        wordCountByTopic[key][word] += 1
    return wordCountByTopic

def getWordTopicCounts(wordCountByTopic):
    wordTopicCounts = collections.Counter()
    for key in wordCountByTopic.keys():
        for word in wordCountByTopic[key].keys():
            wordTopicCounts[word] += 1
    return wordTopicCounts

def createTfIdfVals(wordCountByTopic, wordTopicCounts):
    tfIdfVals = {}
    numTopics = len(wordCountByTopic.keys())
    for key in wordCountByTopic.keys():
        tfIdfVals[key] = collections.Counter()
        for word in wordCountByTopic[key].keys():
            tfIdfVals[key][word] = float(math.log(1.0 + wordCountByTopic[key][word])) * \
                                   float(math.log(1.0 + numTopics/wordTopicCounts[word]))
    return tfIdfVals

def getAvgTfIdf(topic, tfidfvals, words):
    avg = float(sum(tfidfvals[topic][word] for word in words)/len(words))
    print avg, words, [tfidfvals[topic][word] for word in words]
    return avg