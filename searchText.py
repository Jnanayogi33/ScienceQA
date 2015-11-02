#Scrape text from wikipedia and other texts

import wikipedia as wiki
import re
import exceptions
import QAUtils as util

def getKeywords(file):

    keywords = []
    keyWordFile = open(file).readlines()
    for line in keyWordFile[1:]:
        keywords += [line.strip('\n')]

    termlist = []
    for i, keyword in enumerate(keywords):

        print "Getting", str(i+1), "/ " + str(len(keywords)) + " search term lists from Wikipedia"
        tries = 0
        while True:
            try: terms = wiki.search(keyword)
            except: pass
            break
        termlist += terms

    return list(set(termlist))

def createCompendium(keywords, startIndex = 0):

    if startIndex == 0: compendium = {}
    else: compendium = util.loadData('createCompendiumTemp')

    for i, keyword in enumerate(keywords[startIndex:]):
        print "Downloading", str(i+1), "/ " + str(len(keywords)) + " files from Wikipedia"

        tries = 0
        while True:
            try:
                content = wiki.page(keyword)
                text = content.content.encode('ascii', 'ignore')
                title = content.title.encode('ascii', 'ignore')
                subtitle = title
            except: pass
            break

        compendium[title] = {}
        compendium[title][subtitle] = []

        for line in text.split('\n'):
            if len(line) == 0: continue
            match = re.match(r'=+ (.+)Edit =+', line)
            if match != None:
                subtitle = match.group(1)
                if subtitle in ['Notes', 'See also', 'References', 'External links']: break
                compendium[title][subtitle] = []
            else: compendium[title][subtitle] += [line]

        util.saveData(compendium, 'createCompendiumTemp')

    return compendium
