from apiclient.discovery import build
from nltk.corpus import wordnet as wn
from spacy import attrs
import urllib,urllib.request, urllib.error
import json, re
import sys, os
import pickle
import spacy.en
import time, string, math

###############################################################
# Setup

print('Initializing spacy...')
nlp = spacy.en.English()
print('Done!')

# api_key = 'AIzaSyDws9BTSh164jnRU-bCaOPzbycvfLZAvbM'
search_engine_id = '017856859473145577022:dswlvnrydbq'
api_key = 'AIzaSyDhCTJt6qh5UkH-t_p8_M2wZAI07NFNV_Y'
queryResultLimit = 5

# load FB from local store
if os.path.isfile('FB_relations.p'):
	print('FB Relations Found.')
	local = open('FB_relations.p', 'rb')
	freebaseRelations = pickle.load(local)
	local.close()
else:
	freebaseRelations = {}

# load WN from local store
if os.path.isfile('WN_relations.p'):
	print('WN Relations Found.')
	local = open('WN_relations.p', 'rb')
	wnRelations = pickle.load(local)
	local.close()
else:
	wnRelations = {}
###############################################################
# Utility Functions

def getGoogleSnippets(q):
	'''Returns top 20 google snippets for search term q'''
	print('Searching for google snippets for query:', q)
	search_term = q
	service = build('customsearch', 'v1', developerKey=api_key)
	collection = service.cse()

	snippetDoc = []

	for j in range(2):
		request = collection.list(q=search_term, num=10, start= 1 + 10 * j, cx=search_engine_id)
		response = request.execute()
		searchResults = json.dumps(response, sort_keys=True, indent=2)
		searchObject = json.loads(searchResults)
		items = searchObject['items']
		for i in items:
			snippetDoc.append(i['snippet'])
	return ' '.join(snippetDoc)

def getSearchFromFile():
	'''Opens local copy of search results'''
	searchResults = pickle.load(open( "searchResults.p", "rb"))
	searchObject = json.loads(searchResults)
	snippetDoc = ''
	items = searchObject['items']
	for i in items:
		snippetDoc += i['snippet']
	return snippetDoc

def getKeywords(questionText):
	'''Returns array of unique keywords given |text| '''
	question = nlp(questionText)
	is_keyword = lambda t: (not t.is_stop) and t.is_alpha
	keywords = [t.lemma_ for t in question if is_keyword(t) and len(t) > 0]
	return list(set(keywords))

def getToken(word):
	'''Returns spacy token from |word|'''
	w = nlp(word)
	return w[0]

def averageSimilarity(keywords, a):
	answer = getToken(a)
	keywords = [getToken(kw).similarity(answer) for kw in keywords]
	average = sum(keywords) / len(keywords)
	return average

def getImportanceDict(questionText):
	'''Returns dict{word: importance} from |text|'''
	importanceDict = {}
	snippets = nlp(getGoogleSnippets(questionText))
	wordCounts = snippets.count_by(attrs.ORTH)
	keywords = getKeywords(questionText)

	# importance = wordCount / probOfWord
	for kw in keywords:
		kw = getToken(kw)
		if kw.orth in wordCounts:
			value = wordCounts[kw.orth] / math.exp(kw.prob)
		else:
			value = 1 / math.exp(kw.prob)
		importanceDict[kw.orth_] = value

	# normalize
	total = sum(importanceDict.values())
	for kw in importanceDict:
		importanceDict[kw] = importanceDict[kw]/total
	return importanceDict

def getWords(s):
	regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
	words = regex.sub(' ', s).strip().split()
	words = [getToken(w).lemma_.lower() for w in words if w != '']
	return words

def getWNRelations(w):
	print('Getting WordNet relations for word: ', w)
	if w in wnRelations:
		print('Accessed from local store!')
		return wnRelations[w]

	neighbors = []
	for synset in wn.synsets(w):
		hypernyms = synset.hypernyms()
		for h in hypernyms:
			for l in h.lemmas():
				full_name = getWords(l.name())
				for word in full_name:
					# not the same word, not empty string
					if word != w.lower() and len(word) > 0: 
						neighbors.append(word)

	# get rid of duplicates
	neighbors = list(set(neighbors))

	# save locally
	wnRelations[w] = neighbors
	local = open('WN_relations.p', 'wb')
	pickle.dump(wnRelations, local)
	local.close()
	print('WN relations saved:', len(neighbors))

	return neighbors

def topicQuery(topic_id):
	'''# Usage: Must provide a valid freebase MID. For example 
	to get San Francisco triples: topicQuery('/m/0d6lp')'''
	
	if topic_id[0:3] != '/m/': return [] # check for valid topic_id:

	service_url = 'https://www.googleapis.com/freebase/v1/topic'
	params = {
	    'lang' : 'en',
	    'key': api_key,
	    '&limit': queryResultLimit
	    # 'filter': 'suggest',

	}
	url = service_url + topic_id + '?' + urllib.parse.urlencode(params)
	try:
		response = urllib.request.urlopen(url)
	except urllib.error.URLError as e:
		print('URL Error:', e.reason)
		return []
	except urllib.error.HTTPError as e:
		print('HTTP Error code:', e.code)
		return []
	else:
		topic = json.loads(urllib.request.urlopen(url).read().decode())
		if '/type/object/name' not in topic['property']: return []
		name = topic['property']['/type/object/name']['values'][0]['value']

		triples = []
		for property in topic['property']:
		    if property in ['/common/topic/article','/common/topic/image',
		                    '/type/object/name', '/common/topic/topic_equivalent_webpage',
		                    '/type/object/key', '/common/topic/topical_webpage',
		                    '/type/object/mid', '/type/object/guid', '/type/object/permission',
		                    '/type/object/creator', '/common/topic/description', '/type/object/timestamp', '/type/object/attribution']: continue
		    if re.match(r'.*/data_source\b', property) or \
		            re.match(r'.*/all_permission\b', property) or \
		            re.match(r'.*/official_website\b', property) or \
		            re.match(r'/tv/.*', property) or \
		            re.match(r'/book/.*', property) or \
		            re.match(r'/fictional_universe/.*', property) or \
		            re.match(r'/music/.*', property) or \
		            re.match(r'/media.*', property) or \
		            re.match(r'/award/.*', property) or \
		            re.match(r'/film.*', property) or \
		            re.match(r'/business.*', property) or \
		            re.match(r'/sports.*', property) or \
		            re.match(r'/soccer.*', property) or \
		            re.match(r'/cvg.*', property): continue
		    # print(property)
		    for value in topic['property'][property]['values']:
		        if 'lang' in value.keys():
		            if value['lang'] != 'en': continue
		        if 'id' in value.keys(): mid = value['id']
		        else: mid = None
		        triples += [[[name, "Has property " + property, value['text']], mid]]
		return triples

def searchQuery(query):
	'''Usage: Provide it with a string query such as 
	"photosynthesis" and it returns list of search result MIDs '''
	service_url = 'https://www.googleapis.com/freebase/v1/search'
	params = {
	        'query' : query,
	        'lang' : 'en',
	        'key': api_key,
	        '&limit' : queryResultLimit
	}

	url = service_url + '?' + urllib.parse.urlencode(params)
	# print(url)
	response = json.loads(urllib.request.urlopen(url).read().decode())

	mids = []
	for result in response['result']:
	    # print(result)
	    if 'mid' in result.keys():
	        mids += [result['mid']]
	return mids

def getFBRelations(w):
	print('Getting FB relations for word: ', w)
	# check for memory, else get new relations
	if w in freebaseRelations:
		print('Accessed from local store!')
		return freebaseRelations[w]

	neighbors = []
	mids = searchQuery(w)
	for mid in mids:
		triples = topicQuery(mid)
		for triple in triples:
			t, _ = triple
			# print(triple)
			# Check if words are same as |query|
			words = getWords(t[0]) + getWords(t[2])
			for word in words:
				# exclude blanks strings
				if word != w.lower() and len(word) > 0:
					neighbors.append(word)

	# get rid of duplicates
	neighbors = list(set(neighbors))

	# Save relations to memory
	freebaseRelations[w] = neighbors
	local = open('FB_relations.p', 'wb')
	pickle.dump(freebaseRelations, local)
	local.close()
	print('FB relations saved:', len(neighbors))

	return neighbors

def getRelations(w):
	print('Getting relations for word: ', w)
	freebase = getFBRelations(w)
	wordnet = getWNRelations(w)
	neighbors = freebase + wordnet
	return list(set(neighbors))

def getFocus(question):
	'''(questionText, answer, correctAnswer) => Returns keyword 
	in questionText that is the lexical focus of the question'''
	questionText, answers, correctAnswer = question
	keywords = getKeywords(questionText)

	# Hypothesis: lexical focus has shortest w2v distance 
	# from answers
	def similarityToAnswers(w):
		distance = 0
		word = getToken(w)
		for a in answers:
			answer = getToken(a)
			distance += answer.similarity(word)
		return distance

	focus = max(keywords, key=similarityToAnswers)
	return focus

def getLexicalFocus(question):
	questionText, answers, correctAnswer = question
	keywords = getKeywords
	pass

