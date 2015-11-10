#Code is working for python 3.4.3

import re, string

def convertANPNTuple(line):
    match = re.search(r'\b(\w+) \(anpn "(.+)" "(.+)" "(.+)" "(.+)"\)', line)
    if match != None: return [["Can be " + " ".join([match.group(2), match.group(4)]), match.group(3), match.group(5)], int(match.group(1))]
    else:
        match = re.search(r'\b(\w+) \(anpn "(.+)" "(.+)" (NIL) "(.+)"\)', line)
        return [[match.group(3), "Can be " + match.group(2), match.group(5)], int(match.group(1))]

def convertQNTuple(line):
    match = re.search(r'\b(\w+) \(qn "(.+)" "(.+)"\)', line)
    return [[match.group(3), "Measured in", match.group(2)], int(match.group(1))]

def convertANTuple(line):
    match = re.search(r'\b(\w+) \(an "(.+)" "(.+)"\)', line)
    return [[match.group(3), "Can be", match.group(2)], int(match.group(1))]

def convertNNTuple(line):
    match = re.search(r'\b(\w+) \(nn "(.+)" "(.+)"\)', line)
    return [[match.group(2), "There can be", match.group(3)], int(match.group(1))]

def convertNPNTuple(line):
    match = re.search(r'\b(\w+) \(npn "(.+)" "(.+)" "(.+)"\)', line)
    return [[match.group(2), "Can be " + match.group(3), match.group(4)], int(match.group(1))]

def convertNVNPNTuple(line):
    match = re.search(r'\b(\w+) \(nvnpn "(.+)" "(.+)" "(.+)" "(.+)" "(.+)"\)', line)
    if match != None: return [[match.group(2), "Can " + " ".join([match.group(3), match.group(5), match.group(6)]), match.group(4)], int(match.group(1))]
    else:
        match = re.search(r'\b(\w+) \(nvnpn "(.+)" "(.+)" "(.+)" (NIL) "(.+)"\)', line)
        if match != None: return [[match.group(2), "Can " + " ".join([match.group(3), "in/at/on", match.group(6)]), match.group(4)], int(match.group(1))]
        else:
            match = re.search(r'\b(\w+) \(nvnpn "(.+)" "(.+)" "(.+)" "(.+)" (NIL)\)', line)
            if match != None: return [[match.group(2), "Can " + " ".join([match.group(3), match.group(5), "something"]), match.group(4)], int(match.group(1))]
            else:
                match = re.search(r'\b(\w+) \(nvnpn "(.+)" "(.+)" "(.+)" (NIL) (NIL)\)', line)
                return [[match.group(2), "Can " + " ".join([match.group(3), "in/at/on", "something"]), match.group(4)], int(match.group(1))]

def convertNVNTuple(line):
    match = re.search(r'\b(\w+) \(nvn "(.+)" "(.+)" "(.+)"\)', line)
    return [[match.group(2), "Can " + match.group(3), match.group(4)], int(match.group(1))]

def convertNVPNTuple(line):
    match = re.search(r'\b(\w+) \(nvpn "(.+)" "(.+)" "(.+)" "(.+)"\)', line)
    if match != None: return [[match.group(2), "Can " + " ".join([match.group(3), match.group(4)]), match.group(5)], int(match.group(1))]
    else:
        match = re.search(r'\b(\w+) \(nvpn "(.+)" "(.+)" (NIL) "(.+)"\)', line)
        if match != None: return [[match.group(2), "Can " + " ".join([match.group(3), "in/at/on"]), match.group(5)], int(match.group(1))]
        else:
            match = re.search(r'\b(\w+) \(nvpn "(.+)" "(.+)" "(.+)" (NIL)\)', line)
            if match != None: return [[match.group(2), "Can " + " ".join([match.group(3), match.group(4)]), "something"], int(match.group(1))]
            else:
                match = re.search(r'\b(\w+) \(nvpn "(.+)" "(.+)" (NIL) (NIL)\)', line)
                return [[match.group(2), "Can " + " ".join([match.group(3), "in/at/on"]), "something"], int(match.group(1))]

def convertNVTuple(line):
    match = re.search(r'\b(\w+) \(nv "(.+)" "(.+)"\)', line)
    return [[match.group(2), "Can", match.group(3)], int(match.group(1))]

def convertVNPNTuple(line):
    match = re.search(r'\b(\w+) \(vnpn "(.+)" "(.+)" "(.+)" "(.+)"\)', line)
    if match != None: return [[match.group(3), "Can " + " ".join([match.group(2), match.group(4)]), match.group(5)], int(match.group(1))]
    else:
        match = re.search(r'\b(\w+) \(vnpn "(.+)" "(.+)" (NIL) "(.+)"\)', line)
        if match != None: return [[match.group(3), "Can " + " ".join([match.group(2), "in/at/on"]), match.group(5)], int(match.group(1))]
        else:
            match = re.search(r'\b(\w+) \(vnpn "(.+)" "(.+)" "(.+)" (NIL)\)', line)
            if match != None: return [[match.group(3), "Can " + " ".join([match.group(2), match.group(4)]), "something"], int(match.group(1))]
            else:
                match = re.search(r'\b(\w+) \(vnpn "(.+)" "(.+)" (NIL) (NIL)\)', line)
                return [[match.group(3), "Can " + " ".join([match.group(2), "in/at/on"]), "something"], int(match.group(1))]

def convertVNTuple(line):
    match = re.search(r'\b(\w+) \(vn "(.+)" "(.+)"\)', line)
    return [[match.group(3), "Can be", match.group(2)], int(match.group(1))]

def convertVPNTuple(line):
    match = re.search(r'\b(\w+) \(vpn "(.+)" "(.+)" "(.+)"\)', line)
    if match != None: return [[match.group(2), "Can be " + match.group(3), match.group(4)], int(match.group(1))]
    else:
        match = re.search(r'\b(\w+) \(vpn "(.+)" (NIL) "(.+)"\)', line)
        if match != None: return [[match.group(2), "Can be " + "in/at/on", match.group(4)], int(match.group(1))]
        else:
            match = re.search(r'\b(\w+) \(vpn "(.+)" "(.+)" (NIL)\)', line)
            if match != None: return [[match.group(2), "Can be " + match.group(3), "something"], int(match.group(1))]
            else:
                match = re.search(r'\b(\w+) \(vpn "(.+)" (NIL) (NIL)\)', line)
                return [[match.group(2), "Can be " + "in/at/on", "something"], int(match.group(1))]

def getFileLines(file, target=[]):
    lines = []
    for line in open(file).readlines():
        write = True
        for word in target:
            if word.lower() not in line.lower():
                write = False
                break
        if write: lines += [line.strip('\n')]
    return lines

def getTuples(file, converter, target=[]):
    return [converter(line) for line in getFileLines(file, target)]

# Usage: getTuplesMatchingWords('C:/Users/Tin Yun/Desktop/DART database', ['respiration', 'aerobic'])
def getTuplesMatchingWords(dartLocation, searchWords):
    tuples = {}
    tuples['DART-ANPN'] = getTuples(dartLocation + '/anpn-tuples.txt', convertANPNTuple, target=searchWords)
    tuples['DART-QN'] = getTuples(dartLocation + '/qn-tuples.txt', convertQNTuple, target=searchWords)
    tuples['DART-AN'] = getTuples(dartLocation + '/an-tuples.txt', convertANTuple, target=searchWords)
    tuples['DART-NN'] = getTuples(dartLocation + '/nn-tuples.txt', convertNNTuple, target=searchWords)
    tuples['DART-NPN'] = getTuples(dartLocation + '/npn-tuples.txt', convertNPNTuple, target=searchWords)
    tuples['DART-NVNPN'] = getTuples(dartLocation + '/nvnpn-tuples.txt', convertNVNPNTuple, target=searchWords)
    tuples['DART-NVN'] = getTuples(dartLocation + '/nvn-tuples.txt', convertNVNTuple, target=searchWords)
    tuples['DART-NVPN'] = getTuples(dartLocation + '/nvpn-tuples.txt', convertNVPNTuple, target=searchWords)
    tuples['DART-NV'] = getTuples(dartLocation + '/nv-tuples.txt', convertNVTuple, target=searchWords)
    tuples['DART-VNPN'] = getTuples(dartLocation + '/vnpn-tuples.txt', convertVNPNTuple, target=searchWords)
    tuples['DART-VN'] = getTuples(dartLocation + '/vn-tuples.txt', convertVNTuple, target=searchWords)
    tuples['DART-VPN'] = getTuples(dartLocation + '/vpn-tuples.txt', convertVPNTuple, target=searchWords)
    return tuples

print(getTuplesMatchingWords('C:/Users/Tin Yun/Desktop/DART database', ['protein']))