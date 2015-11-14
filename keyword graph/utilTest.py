import util
from Models import Test

def testGetFocus():
	fourthGradeExam = Test(start=0, end=10, N=6)
	test = fourthGradeExam.test

	for question in test:
		questionText, answers, correctAnswer = question
		focus = util.getFocus(question)
		print('Question: {} | Focus: {}'.format(questionText, focus))

def testGetRelations(w):
	neighbors = util.getRelations(w)
	print(len(neighbors))
	return neighbors

# relations = util.getFBRelations('energy')
# for r in relations:
# 	print(r)

def testGetWords(w):
	return util.getWords(w)

# print(util.getRelations('force'))