import util
import pickle
import copy
import time
import os

regentsDataPath = 'Regents_Train.tsv'

class WordGraph:
	def __init__(self, question, N):
		print('Question:', question)
		self.graph = {}
		self.N = N
		self.question_keywords = util.getKeywords(question)
		print('Question keywords extracted:', self.question_keywords)
		self.importance = {kw: 1/len(self.question_keywords) for kw in self.question_keywords}
		# self.importance = util.getImportanceDict(question)
		print('Keyword importance:', self.importance)
		# for kw in self.question_keywords:
		# 	self.addWord(kw, iskeyword=True)
		initialWords = self.bestWords() + self.question_keywords
		print('Nodes are:', initialWords)
		for word in initialWords:
			self.addWord(word)
	
	def addWord(self, w):
		''' Takes in word |w| and adds it to current graph,
		making all appropriate links to existing graph '''
		if w in self.graph:
			print('{} already in graph with {} connections'.format(w, len(self.graph[w])))
		if w not in self.graph: self.graph[w] = []

		possibleNeighbors = util.getRelations(w)
		counter = 0
		for node in self.graph:
			if node in possibleNeighbors:
				# Add relations only if they don't already exist
				if node not in self.graph[w]:
					self.graph[w] += [node]
					counter += 1
				if w not in self.graph[node]:
					self.graph[node] += [w]
		print('Number of new connections:', counter)

	def removeWord(self, w):
		''' Removes |w| from graph by deleting node and all
		connections '''
		if w not in self.graph: return
		# remove node from graph
		neighbors = self.graph.pop(w)
		# remove all connections to the node
		for nbr in neighbors:
			if w in self.graph[nbr]:
				self.graph[nbr].remove(w)

	def relevanceScore(self, w):
		'''Returns score = SUM(kw_importance * similarity)'''
		word = util.getToken(w)
		total = 0
		for kw in self.question_keywords:
			keyword = util.getToken(kw)
			total += self.importance[kw] * keyword.similarity(word)
		return total

	def bestWords(self):
		'''Returns best N * num_keywords 1st order links 
		to question keywords'''
		words = []
		for keyword in self.question_keywords:
			neighbors = util.getRelations(keyword)
			for nbr in neighbors:
				if nbr not in words:
					words.append(nbr)
		words = sorted(words, key=self.relevanceScore)
		words.reverse()
		limit = len(self.question_keywords) * self.N
		return words[0 : limit + 1]

	def coherenceScore(self, w):
		total = 0
		neighbors = self.graph[w]
		word = util.getToken(w)
		for nbr in neighbors:
			total += word.similarity(util.getToken(nbr))
		return total

	def pruneGraph(self):
		'''Prunes graph based on coherence'''
		totalWords = len(self.graph.keys())
		counter = 0
		isValid = lambda word: (word not in self.question_keywords) or (word == self.answer)
		words = [w for w in self.graph if isValid(w)]
		words = sorted(words, key=self.coherenceScore)
		words.reverse()
		while words[-1] != self.answer:
			counter += 1
			worstWord = words.pop()
			self.removeWord(worstWord)
		print('{} out of {} words were pruned'.format(counter, totalWords))

	def getAnswerScore(self, answer):
		self.answer = answer
		self.addWord(self.answer)
		self.pruneGraph()
		finalScore = self.coherenceScore(answer)
		print('For answer {}, final score is {}'.format(answer, finalScore))
		return finalScore

class Test:
	def __init__(self, start, end, N):
		self.LETTERS = ['A', 'B', 'C', 'D']
		self.fullTest = self.getTest()
		self.test = [q for i, q in enumerate(self.fullTest) if (i < end and i >= start)]
		self.correct = 0
		self.incorrect = 0
		self.answerReport = []
		self.timeReport = []
		self.N = N

		# instantiate mindmaps
		if os.path.isfile('mindmaps.p'):
			print('Local Mindmaps found.')
			local = open('mindmaps.p', 'rb')
			self.mindmaps = pickle.load(local)
			local.close()
		else:
			self.mindmaps = {}

	def reset(self):
		self.answerReport, self.timeReport = [], []
		self.correct, self.incorrect = 0, 0

	def getTest(self):
		'''Returns all one-word-answer questions'''
		qa = open(regentsDataPath).readlines()
		test = []
		questionAnswerDict = {}
		questionCorrectAnswerDict = {}

		for i, line in enumerate(qa):
			if i == 0: continue
			sentence = line.split('\t')[9]
			# extract answers
			answers = []
			questionText = sentence.split('(A)')[0]
			optionA = sentence.split('(A)')[1].split('(B)')[0]
			optionB = sentence.split('(A)')[1].split('(B)')[1].split('(C)')[0]
			optionC = sentence.split('(A)')[1].split('(B)')[1].split('(C)')[1].split('(D)')[0]
			answers = [optionA.strip(), optionB.strip(), optionC.strip()]
			if sentence.find('(D)') != -1:
				optionD = sentence.split('(D)')[1]
				answers.append(optionD.strip())

			# get only one word answers
			oneWordAnswers = True
			for a in answers:
				if len(a.split()) > 1:
					oneWordAnswers = False
					break
			if oneWordAnswers:
				correctAnswer = line.split('\t')[3]
				question = (questionText.strip(), answers, correctAnswer)
				test.append(question)
		return test


	def takeTestMindmaps(self):
		self.reset()
		for num, question in enumerate(self.test):
			print('\nQuestion {} ---------------------------'.format(num+1))
			# Think about question -> Generate scene
			start = time.time()
			questionText, answers, correctAnswer = question

			# save mindmap for question
			if questionText in self.mindmaps:
				print('Mindmap accessed from local store!')
				wordGraph = self.mindmaps[questionText]
			else:
				wordGraph = WordGraph(questionText, self.N)
				self.mindmaps[questionText] = wordGraph
				local = open('mindmaps.p', 'wb')
				pickle.dump(self.mindmaps, local)
				local.close()
				print('Mindmap saved.')

			# Compare answer to scene
			answerScores = []
			for ans in answers:
				questionGraph = copy.deepcopy(wordGraph)
				score = questionGraph.getAnswerScore(ans)
				answerScores.append(score)

			# Mark question
			index = answerScores.index(max(answerScores))
			if self.LETTERS[index] == correctAnswer:
				self.correct += 1
			else:
				self.incorrect += 1
			end = time.time()

			self.answerReport.append((answerScores, index, correctAnswer))
			self.timeReport.append(end - start)

	def takeTestW2V(self):
		self.reset()
		for num, question in enumerate(self.test):
			start = time.time()

			questionText, answers, correctAnswer = question
			keywords = util.getKeywords(questionText)

			# Get scores for each answer
			answerScores = []
			for answer in answers:
				answerScores.append(util.averageSimilarity(keywords, answer))

			index = answerScores.index(max(answerScores))
			if self.LETTERS[index] == correctAnswer:
				self.correct += 1
			else:
				self.incorrect += 1
			end = time.time()

			self.answerReport.append((answerScores, index, correctAnswer))
			self.timeReport.append(end - start)



































######################################################
# Tests
# fourthGradeExam = Test.getTest()
# correct, incorrect = Test.takeTest(fourthGradeExam, questions=1)


# question = 'Which force causes rocks to roll downhill'
# wordGraph = WordGraph(question, N=6)
# ans = 'gravity'
# wordGraph.getAnswerScore(ans)
# wordGraph.addWord(ans)
# wordGraph.pruneGraph()




# initialization test
# print(wordGraph.graph)

# getRelations test
# print(wordGraph.getRelations('talk'))

# addNode test
# wordGraph.addWord('test_word')
# print(wordGraph.graph)

# removeNode test
# wordGraph.removeWord('test_word')
# print(wordGraph.graph)