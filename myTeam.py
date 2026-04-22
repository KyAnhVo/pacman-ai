# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random
import time
from asyncio.unix_events import SelectorEventLoop
from enum import Enum
from this import s

import game
import util
from capture import GameState
from captureAgents import CaptureAgent
from game import Directions, Game

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed, first="DummyAgent", second="DummyAgent"):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


def distL1(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def distL1WithEnemyPos(
    a, b, enemyPosBeliefTable: list[list[float]], suicidalness: float
):
    """
    Punishes paths that are potentially close to enemies.

    Args:
        a: The starting position.
        b: The ending position.
        enemyPosBeliefTable: A 2D list representing the belief table of enemy positions.
        suicidalness: A float representing the agent's suicidalness factor.
            Higher values result in more aggressive behavior, i.e. the agent will care less about the enemies.
    """
    dL1 = distL1(a, b)
    dangerFactor = 0.0

    minx = min(a[0], b[0])
    maxx = max(a[0], b[0])
    miny = min(a[1], b[1])
    maxy = max(a[1], b[1])
    area = (maxx - minx) * (maxy - miny)

    for i in range(minx, maxx + 1):
        for j in range(miny, maxy + 1):
            dangerFactor += enemyPosBeliefTable[i][j]
    # This punishes paths that are potentially close to enemies
    return dL1 + dangerFactor * (1 / suicidalness) / area


def closestFoodHeuristic(
    state: GameState,
    agentIndex: int,
    enemyPosBeliefTable: list[list[float]],
    suicidalness: float,
):
    """
    This heuristic returns the distance to the closest food pellet
    """
    pos = state.getAgentPosition(agentIndex)
    if pos is None:
        return float("inf")
    foods = state.getBlueFood() if state.isOnRedTeam(agentIndex) else state.getRedFood()
    closest = float("inf")
    for food in foods:
        distToFood = min(
            closest, distL1WithEnemyPos(pos, food, enemyPosBeliefTable, suicidalness)
        )
        closest = min(closest, distToFood)
    return closest


class GenericPacmanProblem:
    def __init__(self, agent: CaptureAgent, agentIndex: int, startState: GameState):
        self.agent = agent
        self.startState = startState

    def getStartState(self):
        return self.agent.getCurrentObservation()

    def isGoalState(self, gameState: GameState):
        raise NotImplementedError()

    def getSuccessors(self, gameState: GameState):
        legalActions = gameState.getLegalActions(self.agent.index)
        return [
            gameState.generateSuccessor(self.agent.index, action)
            for action in legalActions
        ]


class RushFoodProblem(GenericPacmanProblem):
    def __init__(self, agent: CaptureAgent, agentIndex: int, startState: GameState):
        super().__init__(agent, agentIndex, startState)
        self.foods = self.agent.getFood(self.startState)

    def getStartState(self):
        self.foods = self.agent.getFood(self.startState)

    def isGoalState(self, gameState: GameState):
        pos = gameState.getAgentPosition(self.agent.index)
        return self.foods[pos[0]][pos[1]]

    """
    Defines an agent with attack and defense modes.
    If attack:
        if have much thingy:
            go home
        else:
            if not have capsule:
                if capsule close by:
                    aim to capsule
                else:
                    get point greedy
            if have capsule:
                if capsule timer > CAPSULE_TIMER_WARNING:
                    try to get more thingy
                else:
                    go get thingy but aim to be as far from ghosts as possible

    If defense:
        if not found pacman:
            patrol
        else:
            if not scared:
                run to the pacman
            else:
                keep distance from pacman, dont get zoned

    """

    # problem classes for each state of the agent

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.suicidalness = 0.0


class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.

        """

        """
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        """
        CaptureAgent.registerInitialState(self, gameState)

    """
    Your initialization code goes here, if you need any.
    """

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        """
        You should change this in your own agent.
        """

        return random.choice(actions)
