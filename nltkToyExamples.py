from nltk import *

prover = ResolutionProver()
read_expr = sem.Expression.fromstring

NotFnS = read_expr('-north_of(f, s)')
NotSnF = read_expr('-north_of(s, f)')
SnF = read_expr('north_of(s, f)')
R = read_expr('all x. all y. (north_of(x,y) <-> -north_of(y,x))')
E1 = read_expr('all x. all y. -(north_of(x,y) & north_of(y,x))')
E2 = read_expr('all x. all y. -(-north_of(x,y) & -north_of(y,x))')
print prover.prove(NotFnS, [SnF, R, E1, E2]) # should be true
print prover.prove(NotSnF, [SnF, R, E1, E2]) # should be false

cyrilDog = read_expr('dog(cyril)')
meNotBark = read_expr('-bark(me)')
cyrilBark = read_expr('bark(cyril)')
meDog = read_expr('dog(me)')
dogBark = read_expr('all x. (dog(x) -> bark(x))')
print prover.prove(cyrilBark, [dogBark, cyrilDog]) # should be true
print prover.prove(meDog, [dogBark, meNotBark]) # should be false

kb = []
kb.append(read_expr('tellTruth(john) <-> -crashedServer(john)'))
kb.append(read_expr('tellTruth(susan) <-> crashedServer(nicole)'))
kb.append(read_expr('tellTruth(mark) <-> crashedServer(susan)'))
kb.append(read_expr('tellTruth(nicole) <-> -tellTruth(susan)'))
kb.append(read_expr('exists x. (tellTruth(x) & (all y. (tellTruth(y) -> (x=y))))'))
kb.append(read_expr('exists x. (crashedServer(x) & (all y. (crashedServer(y) -> (x=y))))'))
johnScrewedUp = read_expr('crashedServer(john)')
susanScrewedUp = read_expr('crashedServer(susan)')
nicoleScrewedUp = read_expr('crashedServer(nicole)')
markScrewedUp = read_expr('crashedServer(mark)')
print prover.prove(johnScrewedUp, kb) # should be true
print prover.prove(susanScrewedUp, kb) # should be false
print prover.prove(nicoleScrewedUp, kb) # should be false
print prover.prove(markScrewedUp, kb) # should be false

kb = []
kb.append(read_expr('Parent(john, susan) & Parent(susan, jack)'))
kb.append(read_expr('all x. all y. ((exists k. Parent(x,k) & Parent(k,y)) <-> Grandparent(x,y))'))
grandparent = read_expr('Grandparent(john,jack)')
wrong = read_expr('Grandparent(susan,jack)')
print prover.prove(grandparent, kb) # should be true
print prover.prove(wrong, kb) # should be false

kb = []
kb.append(read_expr('all x. (Even(x) & -Odd(x)) | (Odd(x) & -Even(x))'))
kb.append(read_expr('all x. Even(x) -> (all y. (Successor(x,y) -> Odd(y)))'))
kb.append(read_expr('all x. Odd(x) -> (all y. (Successor(x,y) -> Even(y)))'))
kb.append(read_expr('all x. all y. Successor(x,y) -> Larger(y,x)'))
kb.append(read_expr('all z. all y. all x. (Larger(x,y) & Larger(y,z)) -> Larger(x,z)'))
kb.append(read_expr('all x. (exists y. (Successor(x,y) & ((all z. (Successor(x,z) -> (z=y))) & x!=y)))'))
query = read_expr('all x. (exists y. Even(y) & Larger(y,x))')
print prover.prove(query, kb)