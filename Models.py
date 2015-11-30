import util
import pickle
import copy
import time
import os

regentsDataPath = './ScienceQASharedCache/Regents_Train.tsv'
trainData = './ScienceQASharedCache/training_set.tsv'
validationData = './ScienceQASharedCache/validation_set.tsv'

class WordGraph:
	def __init__(self, question, N):
		# print('Question:', question)
		self.graph = {}
		self.N = N
		self.questionKeywords = util.getKeywords(question)
		# print('Question keywords extracted:', self.questionKeywords)

		self.importance = {kw: 1/len(self.questionKeywords) for kw in self.questionKeywords}
		# self.importance = util.getImportanceDict(question)
		# print('Keyword importance:', self.importance)

		self.secondOrderKeywords = self.bestWords()
		initialWords = self.secondOrderKeywords + self.questionKeywords
		# print('Nodes are:', initialWords)

		for word in initialWords:
			self.addWord(word)
	
	def addWord(self, w):
		''' Takes in word |w| and adds it to current graph,
		making all appropriate links to existing graph '''
		# if w in self.graph:
		# 	print('{} already in graph with {} connections'.format(w, len(self.graph[w])))
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
		# print('Number of new connections:', counter)

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
		for kw in self.questionKeywords:
			keyword = util.getToken(kw)
			total += self.importance[kw] * keyword.similarity(word)
		return total

	def bestWords(self):
		'''Returns best N * num_keywords 1st order links 
		to question keywords'''
		words = []
		for keyword in self.questionKeywords:
			neighbors = util.getRelations(keyword)
			for nbr in neighbors:
				if nbr not in words:
					words.append(nbr)
		words = sorted(words, key=self.relevanceScore)
		words.reverse()
		limit = len(self.questionKeywords) * self.N
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
		# Never remove question keywords, but put answer keywords
		isValid = lambda word: (word not in self.questionKeywords) or (word in self.answerKeywords)
		words = [w for w in self.graph if isValid(w)]
		words = sorted(words, key=self.coherenceScore)
		words.reverse()

		while words[-1] not in self.answerKeywords:
			counter += 1
			worstWord = words.pop()
			self.removeWord(worstWord)
		# print('{} out of {} words were pruned'.format(counter, totalWords))

	def getAnswerScore(self, answer):
		''' |answer| is a string '''

		self.answerKeywords = util.getKeywords(answer)
		if len(self.answerKeywords) == 0:
			return 0
		for keyword in self.answerKeywords:
			self.addWord(keyword)
		self.pruneGraph()

		finalScore = 0
		for keyword in self.answerKeywords:
			finalScore += self.coherenceScore(keyword)
		# print('For answer {}, final score is {}'.format(answer, finalScore))

		# Get search metric
		# searchScore = util.getAnswerCost(self)
		# print('Search score: {}'.format(searchScore))
		return finalScore #, searchScore

	def getNeighbors(self, word):
		return self.graph[word]

	def getSecondOrderKeywords(self):
		return self.secondOrderKeywords

class Test:
	def __init__(self, start, end, level, N):
		self.LETTERS = ['A', 'B', 'C', 'D']
		self.fullTest = self.trainSet() if level == 'eightTrain' else self.getTest()
		self.test = [q for i, q in enumerate(self.fullTest) if (i < end and i >= start)]
		self.correct = 0
		self.incorrect = 0
		self.answerReport = []
		self.searchAnswerReport = []
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

	def trainSet(self):
		test = []
		trainQA = util.extractQA(trainData)
		for i, line in enumerate(trainQA):
			answers = [ans.strip() for ans in line[3:7]]
			qid = line[0]
			questionText = line[1]
			# print('Question text: {}'.format(questionText))
			correctAnswer = line[2]
			question = (questionText.strip(), answers, correctAnswer)
			test.append(question)
		return test

	def validationSet(self):
		test = []
		valRawQA = util.extractQA(validationData, validationSet=True)
		for i, line in enumerate(valRawQA):
			answers = [ans.strip() for ans in line[2:5]]
			qid = line[0]
			questionText = line[1]
			question = (questionText.strip(), answers)
			test.append(question)
		return test


	def getFirstOrderKeywords(self):
		keywords = []
		# for each |question| in full test, get keywords
		for i, question in enumerate(self.fullTest):
			print('Getting keyword for Q{}'.format(i))
			questionText, answers, correctAnswer = question
			keywords += util.getKeywords(questionText)
			for ans in answers:
				keywords += util.getKeywords(ans)
		return list(set(keywords))

	def getSecondOrderKeywords(self):
		keywords = []
		for num, question in enumerate(self.test):
			print('\nQuestion {} ---------------------------'.format(num+1))
			questionText, answers, correctAnswer = question
			wordGraph = WordGraph(questionText, self.N)
			keywords += wordGraph.getSecondOrderKeywords()

		keywords = list(set(keywords))
		print('{} second order keywords found from {} questions'.format(len(keywords), num))
		return keywords

	def takeTest(self):
		self.reset()
		densityCorrect = 0
		searchCorrect = 0
		w2vCorrect = 0
		# Take test
		for num, question in enumerate(self.test):
			print('\nQuestion {} ---------------------------'.format(num+1))
			# Think about question -> Generate scene
			start = time.time()
			questionText, answers, correctAnswer = question

			print('Question: {}'.format(questionText))

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

			keywords = wordGraph.questionKeywords

			# Get density & search scores
			densityScores = []
			# searchScores = []
			# word2vecScores = []
			for ans in answers:
				questionGraph = copy.deepcopy(wordGraph)
				densityScore = questionGraph.getAnswerScore(ans)
				densityScores.append(densityScore)
				# searchScores.append(searchScore)
				# word2vecScores.append(util.averageSimilarity(keywords, ans))

			# Mark using density score
			density_index = densityScores.index(max(densityScores))
			if self.LETTERS[density_index] == correctAnswer:
				self.correct += 1
				densityCorrect += 1
			else:
				self.incorrect += 1

			# Mark question using search scores
			# search_index = searchScores.index(min(searchScores))
			# if self.LETTERS[search_index] == correctAnswer:
			# 	searchCorrect += 1

			# Mark question using word2vec
			# w2v_index = word2vecScores.index(max(word2vecScores))
			# if self.LETTERS[search_index] == correctAnswer:
			# 	w2vCorrect += 1

			end = time.time()

			self.answerReport.append((densityScores, density_index, correctAnswer))
			self.timeReport.append(end - start)

		print('Out of {} questions'.format(len(self.test)))
		print('Density: {}'.format(densityCorrect))

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

