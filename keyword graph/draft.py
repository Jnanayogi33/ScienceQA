someDict = {'jane' : 'rebecaa', 'sa': 'lly'}
metaDict = {}
frozenDict = frozenset(someDict.items())
print(frozenDict)

metaDict[frozenDict] = 1
thawed = dict(frozenDict)

print(metaDict[frozenset(thawed.items())])

# def someFunction(monster):
# 	return monster.monsters

# class Monster:
# 	def __init__(self):
# 		self.monsters = ['Jane']
# 	def trySomething(self):
# 		return someFunction(self)

# monster = Monster()
# print(monster.trySomething())