import countWords as CW

def extractQA(QAfile):
    QAlines = []
    sentences = open(QAfile).readlines()
    for i, sentence in enumerate(sentences[1:]):
        sentences[i] = sentence.split("\t")
        QAlines += [[sentences[i][1], sentences[i][3:], sentences[i][2][0]]]
    return QAlines

def normalizeQA(QAline):
    question = QAline[0]
    answers = QAline[1]
    answerKey = QAline[2]
    question = [CW.standardizeWords(word) for word in question.split()]
    for i, answer in enumerate(answers):
        answers[i] = [CW.standardizeWords(word) for word in answer.split()]
    return question, answers, answerKey