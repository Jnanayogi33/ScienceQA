import os, sys
import QAUtils as utils
from whoosh.fields import *
from whoosh.index import *
from whoosh.query import *
from whoosh.qparser import QueryParser

# 0. Set global parameters
cache = '../Dropbox/ScienceQASharedCache/'

# 1. Get corpus
corpus = utils.loadData(cache + 'allTextLines')[:100]

# 2. Index using whoosh
schema = Schema(content=TEXT, stored_content=TEXT(stored=True))
if not os.path.exists(cache + 'IRindex'):
	os.mkdir(cache + 'IRindex')
ix = create_in(cache + 'IRindex', schema)
ix = open_dir(cache + 'IRindex')

writer = ix.writer()
for i, line in enumerate(corpus):
	sys.stdout.write('\rAdding line {} of {} to index'.format(i+1, len(corpus)))
	sys.stdout.flush()
	writer.add_document(content = line, stored_content = line)
writer.commit()

# Try out a search
with ix.searcher() as searcher:
	query = QueryParser('content', ix.schema).parse('Turkey')
	results = searcher.search(query)
	print(len(results))
	for hit in results:
		print('{}'.format(hit))





