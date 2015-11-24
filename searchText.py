import wikipedia as wiki
import re
import QAUtils as utils
import sys, time
import urllib,urllib.request, urllib.error
import json, re


##################################################################
# Functions for downloading from Wikipedia
##################################################################


# Worker function for downloading search terms from wikipedia. 
#  - If no results then return None
def downloadWikiSearchResults(rawWord):
    terms = wiki.search(rawWord)
    if terms == []: return None
    else: return terms


# Worker function for downloading entire wikipedia page based on the given keyword 
#  - Return contents in dictionary form
def downloadWikiPage(keyword):

    content = wiki.page(keyword)
    text = content.content
    title = content.title
    subtitle = title

    sections = {}
    sections[subtitle] = []
    for line in text.split("\n"):
        if len(line) == 0: continue
        match1 = re.match(r'=+ (.+)Edit =+', line)
        match2 = re.match(r'=+ (.+) =+', line)
        if match1 != None or match2 != None:
            if match1 != None: subtitle = match1.group(1)
            elif match2 != None: subtitle = match2.group(1)
            if subtitle in ['Notes', 'Sources', 'Footnotes', 'See also', 'References', 'External links']: break
            sections[subtitle] = []
        else: sections[subtitle] += [line]
    return sections


# Download all wikipedia pages matching given set of noun chunks
#  - Returns two dictionaries: noun chunk --> keywords, and keyword --> page sections --> list of section paragraphs
#  - Keep separate to minimize memory usage since there would be a lot of redundancy if combined
def getWikipediaCompendium(nounChunks, workerNum = 20, iterations=3, redundancies=False):
    
    print("Getting all wikipedia-specific keywords")
    chunk2keywords = utils.poolDownloader(nounChunks, downloadWikiSearchResults, workerNum, iterations, redundancies)
    utils.saveData(chunk2keywords, 'ScienceQASharedCache/WikiChunk2KeyWords')

    print("Getting all wikipedia pages")
    keywords = []
    for chunk in chunk2keywords.keys(): 
        if chunk2keywords[chunk] == None: continue
        keywords += chunk2keywords[chunk]
    keyword2pages = utils.poolDownloader(keywords, downloadWikiPage, workerNum, iterations, redundancies)

    return chunk2keywords, keyword2pages


##################################################################
# Functions for downloading from Freebase
##################################################################


# Global variables to set first
search_engine_id = '017856859473145577022:dswlvnrydbq'
api_key = 'AIzaSyDhCTJt6qh5UkH-t_p8_M2wZAI07NFNV_Y'
queryResultLimit = 10


# Worker function: given string query, it returns list of search result MIDs
#  - If no search results, return None
def freebaseSearchQuery(query):
    service_url = 'https://www.googleapis.com/freebase/v1/search'
    params = {
            'query' : query,
            'lang' : 'en',
            'key': api_key,
            '&limit' : queryResultLimit
    }

    url = service_url + '?' + urllib.parse.urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read().decode())

    mids = []
    for result in response['result']:
        if 'mid' in result.keys():
            mids += [result['mid']]
    if mids == []: return None
    return mids


# Worker function: given valid freebaseMID returns list of relevant triples
#  - if mid not valid, if result not properly formatted, or there are no triples, return None
#  - Currently filtering out irrelevant information categories, like "fictional universe"
def freebaseTopicQuery(topic_id):
    
    if topic_id[0:3] != '/m/': return None
    service_url = 'https://www.googleapis.com/freebase/v1/topic'
    params = {
        'lang' : 'en',
        'key': api_key,
        '&limit': queryResultLimit
    }
    url = service_url + topic_id + '?' + urllib.parse.urlencode(params)
    topic = json.loads(urllib.request.urlopen(url).read().decode())
    if '/type/object/name' not in topic['property']: return None
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
        for value in topic['property'][property]['values']:
            if 'lang' in value.keys():
                if value['lang'] != 'en': continue
            if 'id' in value.keys(): mid = value['id']
            else: mid = None
            triples += [[[name, "Has property " + property, value['text']], mid]]
    if triples == []: return None
    return triples


# Master function: Download all freebase triples given list of noun chunks
#  - Returns 2 dictionaries: chunk --> list of mids, and mid --> list of triples
#  - Triples in format [[name, "Has property " + property, value['text']], mid of third element in triple]
def getFreebaseCompendium(nounChunks, workerNum = 20, iterations=3, redundancies=False):
    
    print("Getting all freebase MIDs")
    chunk2mids = utils.poolDownloader(nounChunks, freebaseSearchQuery, workerNum, iterations, redundancies)
    utils.saveData(chunk2mids, 'ScienceQASharedCache/FreebaseChunk2Mids')

    print("Getting all freebase triples")
    mids = []
    for chunk in chunk2mids.keys():
        if chunk2mids[chunk] == None: continue
        mids += chunk2mids[chunk]
    mid2triples = utils.poolDownloader(mids, freebaseTopicQuery, workerNum, iterations, redundancies)

    return chunk2mids, mid2triples
