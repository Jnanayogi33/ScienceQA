import wikipedia as wiki
import re
import QAUtils as utils
import sys, time


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


# Master function for downloading search terms from wikipedia
#  - Takes as input list of keywords, returns dictionary mapping original inputs to outputs
def getWikiKeywords(rawWords, workerNum = 20, iterations=3):
    terms = {}
    rawList = utils.workerPool(rawWords, downloadWikiSearchResults, workerNum)
    for result in rawList: terms[result[0]] = result[1]
    return terms


# Master function for getting wikipedia pages given keywords
#  - Takes as input list of keywords, returns dictionary mapping original inputs to outputs
def getWikiPages(keywords, workerNum = 20, iterations=3): 
    pages = {}
    pageList = utils.workerPool(keywords, downloadWikiPage, workerNum, iterations)
    for result in pageList: pages[result[0]] = result[1]
    return pages


# Download all wikipedia pages matching given set of noun chunks
#  - Returns series of dictionaries: noun chunk --> keywords --> page sections --> list of section paragraphs
def getWikipediaCompendium(nounChunks, workerNum = 20, iterations=3):
    
    # print("Getting all wikipedia-specific keywords")
    # chunk2keywords = getWikiKeywords(nounChunks, workerNum, iterations)
    # utils.saveData(chunk2keywords, 'ScienceQASharedCache/WikiChunk2KeyWords')

    chunk2keywords = utils.loadData('ScienceQASharedCache/WikiChunk2KeyWords')
    keywords = []
    for chunk in chunk2keywords.keys(): 
        if chunk2keywords[chunk] == None: continue
        keywords += chunk2keywords[chunk]
    keywords = list(set(keywords))

    print("Getting all wikipedia-specific pages")
    folds = 100
    for i in range(folds):
        lowSplit = int(i*len(keywords)/folds)
        highSplit = int((i+1)*len(keywords)/folds)
        print("Working on pages", lowSplit, "to", highSplit)
        keyword2pages = getWikiPages(keywords[lowSplit:highSplit], workerNum, iterations)
        utils.saveData(keyword2pages, 'ScienceQASharedCache/WikiKeyword2Pages' + str(i))
        time.sleep(60)

    return None
    
    # print("Consolidating wikipedia scraper results")
    # compendium = {}
    # for chunk in chunk2keywords.keys():
    #     compendium[chunk] = {}
    #     if chunk2keywords[chunk] == None:
    #         compendium[chunk] = None
    #         continue
    #     for keyword in chunk2keywords[chunk]:
    #         if keyword in keyword2pages.keys():
    #             compendium[chunk][keyword] = keyword2pages[keyword]

    # return compendium