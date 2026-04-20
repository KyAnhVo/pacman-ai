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
from enum import Enum

import game
import util
from captureAgents import CaptureAgent
from game import Directions

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

        

class MyAgent(CaptureAgent):

    # constants to do state boundaries
    CAPSULE_TIMER_WARNING = 0.5
    MANY_POINTS_THRESHHOLD = 5
    MAX_DIST_CHASE_CAPSULE = 0.1

    class Role(Enum):
        ATTACK = 1
        DEFENSE = 2

    class AttackMode(Enum):
        RUN_HOME = 1
        TO_CAPSULE = 2
        GREEDY_POINT = 3

    class DefendMode(Enum):
        PATROL = 1
        CHASE = 2
        FEARED = 3
        
    '''
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

    '''

    # problem classes for each state of the agent

    class RushFoodProblem:
        def __init__(self, gameState):
            self.start_gameState = gameState

        def getStartState():
            return self.start_gameState

        def isGoalState():
            if gameState.


    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(gameState)


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

    
