import heapq, collections, re, sys, time, os, random

############################################################
# Abstract interfaces for search problems and search algorithms.

class SearchProblem:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return whether |state| is a goal state or not.
    def isGoal(self, state): raise NotImplementedError("Override me")

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state): raise NotImplementedError("Override me")

class SearchAlgorithm:
    # First, call solve on the desired SearchProblem |problem|.
    # Then it should set two things:
    # - self.actions: list of actions that takes one from the start state to a goal
    #                 state; if no action sequence exists, set it to None.
    # - self.totalCost: the sum of the costs along the path or None if no valid
    #                   action sequence exists.
    def solve(self, problem): raise NotImplementedError("Override me")

############################################################
# Uniform cost search algorithm (Dijkstra's algorithm).

class UniformCostSearch(SearchAlgorithm):
    def __init__(self, verbose=0):
        self.verbose = verbose

    def solve(self, problem):
        # If a path exists, set |actions| and |totalCost| accordingly.
        # Otherwise, leave them as None.
        self.actions = None
        self.totalCost = None
        self.numStatesExplored = 0

        # Initialize data structures
        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}  # map state to (action, previous state)

        # Add the start state
        startState = problem.startState()
        frontier.update(startState, 0)

        while True:
            # Remove the state from the queue with the lowest pastCost
            # (priority).
            state, pastCost = frontier.removeMin()
            if state == None: break
            self.numStatesExplored += 1
            if self.verbose >= 2:
                print("Exploring {} with pastCost {}".format(state, pastCost))

            # Check if we've reached the goal; if so, extract solution
            if problem.isGoal(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                if self.verbose >= 1:
                    print("numStatesExplored = {}".format(self.numStatesExplored)) 
                    print('totalCost = {}'.format(self.totalCost))
                    print('actions = {}'.format(self.actions))

                return

            # Expand from |state| to new successor states,
            # updating the frontier with each newState.
            for action, newState, cost in problem.succAndCost(state):
                if self.verbose >= 3:
                    print("  Action {} => {} with cost {} + {}".format(action, newState, pastCost, cost))
                if frontier.update(newState, pastCost + cost):
                    # Found better way to go to |newState|, update backpointer.
                    backpointers[newState] = (action, state)
        if self.verbose >= 1:
            print("No path found")

# Data structure for supporting uniform cost search.
class PriorityQueue:
    def  __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert |state| into the heap with priority |newPriority| if
    # |state| isn't in the heap or |newPriority| is smaller than the existing
    # priority.
    # Return whether the priority queue was updated.
    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority)
    # or (None, None) if the priority queue is empty.
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE: continue  # Outdated priority, skip
            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None) # Nothing left...

class graphSearchProblem(SearchProblem):
    def __init__(self, question, answer, parseGraph, minNewEdges=10):

        self.question = question
        self.answer = answer
        self.parseGraph = parseGraph
        self.minNewEdges = minNewEdges

        self.parseGraph.addQuestion(question, minNewEdges)
        self.parseGraph.addAnswer(answer, minNewEdges)

    def startState(self): return 'question'
    def isGoal(self, state): return state == 'answer'
    def succAndCost(self, state):
        results = []
        for index, node in enumerate(self.parseGraph.getNeighbors(state)):
            results.append([node, node, 1])
        return results

def graphSearchAlgorithm(question, answer, parseGraph, minNewEdges=10):
    ucs = UniformCostSearch(verbose=0)
    ucs.solve(graphSearchProblem(question, answer, parseGraph, minNewEdges))
    shortestPath = ucs.actions
    if shortestPath == None: return float(inf), None
    sequence = [parseGraph.getNodeValue(action) for action in shortestPath]
    parseGraph.removeNode('question')
    parseGraph.removeNode('answer')
    return len(shortestPath), sequence