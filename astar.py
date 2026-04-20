''' this from assignment 1 btw '''

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    initState = problem.getStartState()
    visited = set()

    # parent keep track in a dictionary
    parent = {}
    parent[initState] = (None, None, None)

    # depth keeps track of a node's depth, so that we can look set child's depth
    # to parent's depth + 1.
    depth = {}

    # This is the main queue
    queue = util.PriorityQueue()
    queue.push(initState, 0)

    while not queue.isEmpty():
        currNode = queue.pop()
        #print(currNode)

        # if visited - dont care
        if currNode in visited:
            continue
        visited.add(currNode)

        # check parent, set node's depth to parent's depth
        if parent[currNode][0] != None:
            depth[currNode] = depth[parent[currNode][0]] + 1
        else:
            depth[currNode] = 1

        # check end 
        if problem.isGoalState(currNode):
            # implement reverse dir
            moves = []
            #print(moves)
            while parent[currNode][0] != None:
                moves.append(parent[currNode][1])
                currNode = parent[currNode][0]
            moves.reverse()
            return moves

        # append children
        for child, move, _ in problem.getSuccessors(currNode):
            #print(child, end=" ")
            if child in visited:
                continue
            parent[child] = (currNode, move)
            queue.update(child, 1 / (depth[currNode] + 1))
        #print()
    

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    initState = problem.getStartState()
    visited = set()

    # parent keep track in a dictionary
    parent = {}
    parent[initState] = (None, None)

    # depth keeps track of a node's depth, so that we can look set child's depth
    # to parent's depth + 1.
    depth = {}

    # This is the main queue
    queue = util.PriorityQueue()
    queue.push(initState, 0)

    while not queue.isEmpty():
        currNode = queue.pop()
        #print(currNode)

        # if visited - dont care
        if currNode in visited:
            continue
        visited.add(currNode)

        # check parent, set node's depth to parent's depth
        if parent[currNode][0] != None:
            depth[currNode] = depth[parent[currNode][0]] + 1
        else:
            depth[currNode] = 1

        # check end 
        if problem.isGoalState(currNode):
            # implement reverse dir
            moves = []
            #print(moves)
            while parent[currNode][0] != None:
                moves.append(parent[currNode][1])
                currNode = parent[currNode][0]
            moves.reverse()
            return moves

        # append children
        for child, move, _ in problem.getSuccessors(currNode):
            #print(child, end=" ")
            if child in visited:
                continue
            
            if child in parent:
                childParentDepth = depth[parent[child][0]]
                if childParentDepth < depth[currNode]:
                    continue
            parent[child] = (currNode, move)
            queue.update(child, depth[currNode] + 1)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    initState = problem.getStartState()
    visited = set()

    # current parent, key = state, value = (parent state, move, cost to state if use this path)
    parent = {}
    parent[initState] = (None, None, 0)

    # distance
    dist = {}

    # main queue
    queue = util.PriorityQueue()
    queue.push(initState, 0)

    while not queue.isEmpty():
        currState = queue.pop()
        if currState in visited:
            continue
        visited.add(currState)

        if parent[currState][0] is None:
            dist[currState] = 0
        else:
            dist[currState] = parent[currState][2]

        # check end 
        if problem.isGoalState(currState):
            # implement reverse dir
            moves = []
            #print(moves)
            while parent[currState][0] != None:
                moves.append(parent[currState][1])
                currState = parent[currState][0]
            moves.reverse()
            return moves

        # do children enqueueing
        for child, move, cost in problem.getSuccessors(currState):
            if child in visited:
                continue
            if child in parent:
                _, _, currCost = parent[child]
                if currCost < dist[currState] + cost:
                    continue
            parent[child] = (currState, move, dist[currState] + cost)
            queue.push(child, parent[child][2])
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    initState = problem.getStartState()
    visited = set()

    # current parent, key = state, value = (parent state, move, cost to state if use this path)
    parent = {}
    parent[initState] = (None, None, 0)

    # distance
    dist = {}

    # main queue
    queue = util.PriorityQueue()
    queue.push(initState, 0)

    while not queue.isEmpty():
        currState = queue.pop()
        if currState in visited:
            continue
        visited.add(currState)

        if parent[currState][0] is None:
            dist[currState] = 0
        else:
            dist[currState] = parent[currState][2]

        # check end 
        if problem.isGoalState(currState):
            # implement reverse dir
            moves = []
            #print(moves)
            while parent[currState][0] != None:
                moves.append(parent[currState][1])
                currState = parent[currState][0]
            moves.reverse()
            return moves

        # do children enqueueing
        for child, move, cost in problem.getSuccessors(currState):
            if child in visited:
                continue
            if child in parent:
                _, _, currCost = parent[child]
                if currCost < dist[currState] + cost:
                    continue
            parent[child] = (currState, move, dist[currState] + cost)
            queue.push(child, parent[child][2] + heuristic(child, problem))
    util.raiseNotDefined()



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
