import searchText as scraper
import util
from Models import Test
import pickle, os, time

# Get local copy of freebase
if os.path.isfile('FB_relations.p'):
	print('FB Relations Found.')
	local = open('FB_relations.p', 'rb')
	freebaseRelations = pickle.load(local)
	local.close()
else:
	freebaseRelations = {}

# Setup for worker pool
poolWorkerNum = 200
poolIterations = 2
poolRedundancies = False

# Get all keywords
eightGradeExam = Test(start=0, end=2501, level='eight', N=6)
keywords = eightGradeExam.getSecondOrderKeywords()

# save second order keywords
local = open('SecondOrderKeywords.p', 'wb')
pickle.dump(keywords, local)
local.close()
print('Keywords saved.')

# Filter keywords already in local freebaseRelations
keywords = [kw for kw in keywords if kw not in freebaseRelations]
print('Number of second order keywords left: {}'.format(len(keywords)))

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
local = open('FB_relations.p', 'wb')
pickle.dump(freebaseRelations, local)
local.close()
print('FB relations saved.')

print('Download took {}s'.format(end_download - start_download))
print('Unpacking took {}s'.format(end_unpack - start_unpack))