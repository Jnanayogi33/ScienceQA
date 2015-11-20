import wikipedia as wiki
import re
import QAUtils as util


# Worker function for downloading search terms from wikipedia. 
#  - Returns tuple where first is original search word, second is wiki results
#  - If no results then return None
def downloadWikiSearchResults(rawWord):
    
    tries = 0
    while True:
        try: 
            if tries > 20: break
            else: tries += 1
            terms = wiki.search(rawWord)
        except: pass
        break
    
    try:
        if terms == []: 
            print("No results for noun chunk:", rawWord)
            return None
        else: return (rawWord, terms)
    except:
        print("Problem with noun chunk query:", rawWord)
        return None

# Master function for downloading search terms from wikipedia
#  - Uses default 20 workers because that is max I have found in China that doesn't get blocked
#  - Takes as input list of keywords, returns dictionary mapping original inputs to outputs
def getKeywords(rawWords, workerNum = 20):
    rawList = util.workerPool(rawWords, downloadWikiSearchResults, workerNum)
    terms = {}
    for result in rawList:
        terms[result[0]] = result[1]
    return terms


# Worker function for downloading entire wikipedia page based on the given keyword
#  - Returns tuple where first is original search keyword, second is wiki page in dictionary form
#  - If problem with query, will announce it, and not return anything
#  - pooling function will drop the None when returning final results
def downloadWikiPage(keyword):

    tries = 0
    while True:
        try:
            if tries > 20: break
            else: tries += 1
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
        return (keyword, sections)
    except: 
        print("Problem with keyword query:", keyword)
        return None


# Master function for getting wikipedia pages given keywords
#  - Returns dictionary mapping keywords to their respective pages
def getPages(keywords, workerNum = 20): 
    pageList = util.workerPool(keywords, downloadWikiPage, workerNum)
    pages = {}
    for result in pageList:
        pages[result[0]] = result[1]
    return pages


# Download all wikipedia pages matching given set of noun chunks
#  - Returns series of dictionaries: noun chunk --> keywords --> page sections --> list of section paragraphs
#  - Uses default 20 workers because that is max I have found in China that doesn't get blocked. In US can probably set at 100
def getWikipediaCompendium(nounChunks, workerNum = 20):
    
    print("Getting all wikipedia-specific keywords")
    chunk2keywords = getKeywords(nounChunks, workerNum)
    keywords = []
    for chunk in chunk2keywords.keys(): keywords += chunk2keywords[chunk]
    keywords = list(set(keywords))
    utils.save(chunk2keywords, 'ScienceQASharedCache/WikiChunk2KeyWords')

    print("Getting all wikipedia-specific pages")
    keyword2pages = getPages(keywords, workerNum)

    print("Consolidating wikipedia scraper results")
    compendium = {}
    for chunk in chunk2keywords.keys():
        compendium[chunk] = {}
        for keyword in chunk2keywords[chunk]:
            compendium[chunk][keyword] = None
            if keyword in keyword2pages.keys():
                compendium[chunk][keyword] = keyword2pages[keyword]

    return compendium