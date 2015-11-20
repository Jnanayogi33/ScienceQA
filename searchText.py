import wikipedia as wiki
import re
import QAUtils as util


# Worker function for downloading search terms from wikipedia
def downloadWikiSearchResults(rawWord):
    while True:
        try: terms = wiki.search(rawWord)
        except: pass
        break
    return terms


# Master function for downloading search terms from wikipedia
#  - Uses default 20 workers because that is max I have found in China that doesn't get blocked
#  - takes as input list of keywords, returns list of search terms that API provides
def getKeywords(rawWords, workerNum = 20):
    rawList = util.workerPool(rawWords, downloadWikiSearchResults, workerNum)
    terms = []
    for term in rawList:
        terms += term
    return list(set(terms))


# Worker function for downloading entire wikipedia page based on the given keyword
#  - If problem with query, will announce it, and not return anything
#  - pooling function will drop the None when returning final results
def downloadWikiPage(keyword):

    while True:
        try:
            content = wiki.page(keyword)
            text = content.content
            title = content.title
            subtitle = title
        except: pass
        break

    try:
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
        return [title, sections]
    except: 
        print("Problem with query:", keyword)
        return None


# To be built: master function for downloading all relevant wikipedia pages based on noun chunks