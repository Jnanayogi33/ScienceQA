from logic import *

#Some toy examples to make sure that KB works with basic statements (based off of homework 8) and can do basic resolution tasks

john = Constant('john')
jack = Constant('jack')
jane = Constant('jane')
def Parent(x, y): return Atom('Parent', x, y)
def Grandmother(x, y): return Atom('Grandmother', x, y)
def Grandparent(x, y): return Exists('$k', And(Parent(x,'$k'),Parent('$k',y)))
def Female(x): return Atom('Female', x)
def Male(x): return Atom('Male', x)

kb = createModelCheckingKB()
kb.tell(Parent(john, jack))
kb.tell(Parent(jane, john))
kb.tell(Female(jane))
kb.tell(Forall('$x', Implies(Male('$x'), Not(Female('$x')))))
kb.tell(Forall('$x', Forall('$y', Equiv(And(Grandparent('$x','$y'), Female('$x')), Grandmother('$x','$y')))))
print kb.ask(Grandmother(jane, jack))


def ints():
    def Even(x): return Atom('Even', x)                  # whether x is even
    def Odd(x): return Atom('Odd', x)                    # whether x is odd
    def Successor(x, y): return Atom('Successor', x, y)  # whether x's successor is y
    def Larger(x, y): return Atom('Larger', x, y)        # whether x is larger than y

    formulas = []
    query = Forall('$x', Exists('$y', And(Odd('$y'), Larger('$y', '$x'))))

    formulas.append(Even(Constant('two')))
    formulas.append(Odd(Constant('one')))
    formulas.append(Forall('$x', Or(And(Even('$x'), Not(Odd('$x'))), And(Odd('$x'), Not(Even('$x'))))))
    formulas.append(Forall('$x', Implies(Even('$x'), Forall('$y', Implies(Successor('$x', '$y'), Odd('$y'))))))
    formulas.append(Forall('$x', Implies(Odd('$x'), Forall('$y', Implies(Successor('$x', '$y'), Even('$y'))))))
    formulas.append(Forall('$x', Forall('$y', Implies(Successor('$x', '$y'), Larger('$y', '$x')))))
    formulas.append(Forall('$z', Forall('$y', Forall('$x', Implies(And(Larger('$x', '$y'), Larger('$y', '$z')), Larger('$x', '$z'))))))
    formulas.append(Forall('$x', Exists('$y', AndList([Successor('$x', '$y'), Forall('$z', Implies(Successor('$x', '$z'), Equals('$z', '$y'))), Not(Equals('$x','$y'))]))))

    return (formulas, query)

kb = createModelCheckingKB()
formulas, query = ints()
for formula in formulas:
    kb.tell(formula)
print kb.ask(query)

A = Constant('a')
B = Constant('b')
C = Constant('c')
D = Constant('d')
def tr(x): return Atom('Tr', x)
kb = createModelCheckingKB()
kb.tell(Or(tr(A), tr(B)))
kb.tell(Implies(tr(B),tr(C)))
kb.tell(Implies(Or(tr(A),tr(C)),tr(D)))
print kb.ask(tr(D))

A = Constant('a')
B = Constant('b')
C = Constant('c')
def tr(x): return Atom('Tr', x)
kb = createModelCheckingKB()
kb.tell(Implies(Or(tr(A),tr(B)),tr(C)))
kb.tell(tr(A))
print kb.ask(tr(C))