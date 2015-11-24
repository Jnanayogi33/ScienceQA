#Code is for python 3.4.3

import json, re
import urllib,urllib.request
import os,sys
os.chdir('C:/Users/Tin Yun/Dropbox/Stanford/CS 221/Final Project/ScienceQA')
api_key = open("freebase_api_key.txt").read()
queryResultLimit = 100000

# Usage: Must provide a valid freebase MID. For example to get San Francisco triples: topicQuery('/m/0d6lp')
def topicQuery(topic_id):
    service_url = 'https://www.googleapis.com/freebase/v1/topic'
    params = {
        'lang' : 'en',
        'key': api_key,
        '&limit': queryResultLimit
    }
    url = service_url + topic_id + '?' + urllib.parse.urlencode(params)
    topic = json.loads(urllib.request.urlopen(url).read().decode())
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
        print(property)
        for value in topic['property'][property]['values']:
            if 'lang' in value.keys():
                if value['lang'] != 'en': continue
            if 'id' in value.keys(): mid = value['id']
            else: mid = None
            triples += [[[name, "Has property " + property, value['text']], mid]]

    return triples

# Usage: Provide it with a string query such as "photosynthesis" and it returns list of search result MIDs
def searchQuery(query):
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
        print(result)
        if 'mid' in result.keys():
            mids += [result['mid']]
    return mids


# Get all triples related to the solar system
mids = searchQuery('fission')
for mid in mids:
    triples = topicQuery(mid)
    for triple in triples:
        print(triple)