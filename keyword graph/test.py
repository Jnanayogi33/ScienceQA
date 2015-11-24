from Models import WordGraph, Test
import pickle
import copy
import time
import os
import numpy as np

# Main
start, end = 0, 50
exam = Test(start, end, level=8, N=6)
exam.takeTest()
# exam.takeTestMindmaps()
# numCorrect, numIncorrect = exam.correct, exam.incorrect
# answerReport = exam.answerReport
# timeReport = exam.timeReport

# exam.takeTestW2V()
# benchmarkCorrect, benchmarkInorrect = exam.correct, exam.incorrect
# benchmarkReport = exam.answerReport

# LETTERS = exam.LETTERS
# mindmapMarked = [1 if LETTERS[index] == correctAnswer else -1 for answerScores, index, correctAnswer in answerReport]
# w2vMarked = [1 if LETTERS[index] == correctAnswer else -1 for answerScores, index, correctAnswer in benchmarkReport]

# print('\nREPORTS')
# print('\nAnswer report -----------------------------------------------------')
# for i in range(len(answerReport)):
# 	answerScores, index, correctAnswer = answerReport[i]
# 	benchmarkScores, benchmarkIndex, correctAnswer = benchmarkReport[i]
# 	correct = LETTERS[index] == correctAnswer
# 	mark = lambda b: 'Correct' if b == 1 else 'Wrong'
# 	print('\nQuestion {}: [mindmap] {} [word2vec] {}'.format(i+1, mark(mindmapMarked[i]), mark(w2vMarked[i])))
# 	if not correct:
# 		ourChoice = answerScores[index]
# 		rightChoice = answerScores[LETTERS.index(correctAnswer)]
# 		margin = rightChoice / ourChoice - 1
# 		print('Error margin: {:.1%}'.format(margin))
# 	else:
# 		ourChoice = answerScores[index]
# 		answerScores.pop(index)
# 		score = max(answerScores)
# 		margin = ourChoice / score - 1 
# 		print('Right margin: {:.1%}'.format(margin))

# print('\nScore report ------------------------------------------------------')
# print('[mindmaps] \t Correct: {} \t Incorrect: {}'.format(numCorrect, numIncorrect))
# print('[word2vec] \t Correct: {} \t Incorrect: {}'.format(benchmarkCorrect, benchmarkInorrect))

# mindmapCorrectw2vWrong = sum([1 for i, ans in enumerate(mindmapMarked) if (ans == 1 and w2vMarked[i] == -1)])
# w2vCorrectmindmapWrong = sum([1 for i, ans in enumerate(w2vMarked) if (ans == 1 and mindmapMarked[i] == -1)])
# print('mindmap right, w2v wrong:', mindmapCorrectw2vWrong)
# print('w2v right, mindmap wrong:', w2vCorrectmindmapWrong)
# print('Correlation coefficient:', np.corrcoef(np.array(mindmapMarked), np.array(w2vMarked))[0,1])

# print('\nTime report -----------------------------------------------------')
# averageTime = sum(timeReport)/len(timeReport)
# print('Average time per question: {} min(s) {} s'.format(int(averageTime//60), int(averageTime%60)))



