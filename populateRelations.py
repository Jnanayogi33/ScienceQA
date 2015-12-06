import searchText as scraper
import util
import QAUtils as utils
from Models import Test
import pickle, os, time

cache = '../Dropbox/ScienceQASharedCache/'

# Get local copy of freebase
if os.path.isfile(cache + 'FB_relations.p'): freebaseRelations = utils.loadData(cache + 'FB_relations.p')
else:
	freebaseRelations = {}

# Setup for worker pool
poolWorkerNum = 200
poolIterations = 2
poolRedundancies = False

# Get all keywords
eightGradeExam = Test(start=0, end=8132, dataType='val', N=6)

keywords = eightGradeExam.getSecondOrderKeywords()

# save second order keywords
utils.saveData(keywords, cache + 'SecondOrderKeywords.p')
print('Keywords saved.')

# Filter keywords already in local freebaseRelations
keywords = [kw for kw in keywords if kw not in freebaseRelations]
print('Number of first order keywords left: {}'.format(len(keywords)))

start_download = time.time()

# Get keywords from Freebase
freebaseChunk2Mids, freebaseMid2Triples = scraper.getFreebaseCompendium(keywords, workerNum = poolWorkerNum, iterations=poolIterations, redundancies=poolRedundancies)

end_download = time.time()

start_unpack = time.time()

counter = 0
# Unpack and save
for word, mids in freebaseChunk2Mids.items():
	counter += 1
	print('{} out of {}'.format(counter, len(keywords)))
	if word == '' or mids == None: continue
	if word in freebaseRelations: continue

	neighbors = []
	# get all neighbors for this word
	for mid in mids:
		triples = freebaseMid2Triples[mid]
		neighbors += util.unpackTriples(triples, word)
	
	# eliminate duplicates
	neighbors = list(set(neighbors))

	# add neighbors to local store
	freebaseRelations[word] = neighbors
end_unpack = time.time()

# Save local store
utils.saveData(freebaseRelations, cache + 'FB_relations.p')
print('FB relations saved.')

print('Download took {}s'.format(end_download - start_download))
print('Unpacking took {}s'.format(end_unpack - start_unpack))