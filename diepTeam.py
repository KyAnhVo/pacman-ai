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

import math
import random
import time

import game
import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint

#################
# Team creation #
#################


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="OffensiveMCTSAgent",
    second="DefensiveMCTSAgent",
):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


###########
# Helpers #
###########


def getReverseAction(action):
    if action == Directions.NORTH:
        return Directions.SOUTH
    if action == Directions.SOUTH:
        return Directions.NORTH
    if action == Directions.EAST:
        return Directions.WEST
    if action == Directions.WEST:
        return Directions.EAST
    return action


#############
# MCTS Tree #
#############


class MCTSNode:
    def __init__(
        self, agent, gameState, parent=None, prevAction=None, untriedActions=None
    ):
        self.agent = agent
        self.gameState = gameState
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.value = 0.0
        self.prevAction = prevAction
        all_actions = gameState.getLegalActions(agent.index)
        self.untriedActions = (
            untriedActions
            if untriedActions is not None
            else [a for a in all_actions if a != Directions.STOP]
        )

    def select(node):
        """Descend via UCB1 until we reach a node that needs expansion or is a leaf."""
        exploration_weight = math.sqrt(2)
        current = node
        while current.children and not current.untriedActions:
            current = max(
                current.children,
                key=lambda c: (
                    (c.value / c.visits)
                    + exploration_weight
                    * math.sqrt(math.log(current.visits) / c.visits)
                ),
            )
        return current

    def expand(node):
        """Add one new child for an untried action."""
        if not node.untriedActions:
            return node
        action = node.untriedActions.pop()
        nextState = node.gameState.generateSuccessor(node.agent.index, action)

        # Prune reverse of this action from the child's untried list (unless it's the only move)
        reverse = getReverseAction(action)
        all_actions = [
            a
            for a in nextState.getLegalActions(node.agent.index)
            if a != Directions.STOP
        ]
        if reverse in all_actions and len(all_actions) > 1:
            all_actions.remove(reverse)

        child = MCTSNode(
            agent=node.agent,
            gameState=nextState,
            parent=node,
            prevAction=action,
            untriedActions=all_actions,
        )
        node.children.append(child)
        return child

    def simulate(node):
        currentState = node.gameState
        prevAction = node.prevAction
        visited_positions = {}
        depth = 0

        while not currentState.isOver() and depth < 10:
            actions = [
                a
                for a in currentState.getLegalActions(node.agent.index)
                if a != Directions.STOP
            ]

            reverse = getReverseAction(prevAction) if prevAction else None
            if reverse and reverse in actions and len(actions) > 1:
                actions.remove(reverse)

            if not actions:
                return node.agent.evaluate(currentState, Directions.STOP)

            if random.random() < 0.8:

                def score_action(a):
                    successor = currentState.generateSuccessor(node.agent.index, a)
                    pos = successor.getAgentPosition(node.agent.index)
                    revisit_penalty = visited_positions.get(pos, 0) * 5.0
                    features = node.agent.getFeatures(successor, Directions.STOP)
                    weights = node.agent.getWeights(successor, Directions.STOP)
                    return features * weights - revisit_penalty

                action = max(actions, key=score_action)
            else:
                action = random.choice(actions)

            currentState = currentState.generateSuccessor(node.agent.index, action)
            pos = currentState.getAgentPosition(node.agent.index)
            visited_positions[pos] = visited_positions.get(pos, 0) + 1
            prevAction = action
            depth += 1

        return node.agent.evaluate(currentState, Directions.STOP)

    def backpropagate(node, result):
        """Accumulate raw result as wins; value is the running mean."""
        node.visits += 1
        node.wins += result
        node.value = node.wins / node.visits
        if node.parent:
            MCTSNode.backpropagate(node.parent, result)


####################
# MCTS Base Agent  #
####################


class MCTSCaptureAgent(CaptureAgent):
    """
    Base class for MCTS agents. Subclasses implement getFeatures() and
    getWeights() exactly as in baselineTeam.py. The evaluate() method is
    inherited and used both for final action selection and MCTS scoring.
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        allFood = len(self.getFood(gameState).asList())
        self.foodThreshold = allFood / 2
        mid_x = gameState.data.layout.width // 2
        patrolX = mid_x - 1 if self.red else mid_x
        self.patrolPoints = [
            (patrolX, y)
            for y in range(gameState.data.layout.height)
            if not gameState.hasWall(patrolX, y)
        ]
        self.recentPositions = []

    def getSuccessor(self, gameState, action):
        """Finds the next successor which is a grid position (mirrors baseline helper)."""
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            legalActions = successor.getLegalActions(self.index)
            if action in legalActions:
                return successor.generateSuccessor(self.index, action)
        return successor

    def evaluate(self, gameState, action):
        """Computes a linear combination of features and feature weights."""
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features["successorScore"] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {"successorScore": 1.0}

    def chooseAction(self, gameState):
        # Track recent positions for oscillation detection
        my_pos = gameState.getAgentPosition(self.index)
        self.recentPositions.append(my_pos)
        if len(self.recentPositions) > 6:
            self.recentPositions.pop(0)

        root = MCTSNode(agent=self, gameState=gameState)
        start = time.time()

        while time.time() - start < 0.5:
            node = MCTSNode.select(root)
            if node.untriedActions:
                node = MCTSNode.expand(node)
            result = MCTSNode.simulate(node)
            MCTSNode.backpropagate(node, result)

        if not root.children:
            legal = [
                a for a in gameState.getLegalActions(self.index) if a != Directions.STOP
            ]
            return random.choice(legal or gameState.getLegalActions(self.index))

        # Oscillation guard: if bouncing A->B->A->B, prefer children that lead somewhere fresh
        oscillating = (
            len(self.recentPositions) >= 4
            and self.recentPositions[-1] == self.recentPositions[-3]
            and self.recentPositions[-2] == self.recentPositions[-4]
        )
        if oscillating:
            best_visits = max(c.visits for c in root.children)
            candidates = [c for c in root.children if c.visits >= best_visits * 0.5]
            fresh = [
                c
                for c in candidates
                if c.gameState.getAgentPosition(self.index)
                not in self.recentPositions[-4:]
            ]
            if fresh:
                return max(fresh, key=lambda c: c.visits).prevAction

        return max(root.children, key=lambda c: c.visits).prevAction

    def getEnemyPositions(self, gameState, enemy_indices):
        """
        Triangulates enemy positions and returns the Manhattan distance
        to the closest possible tile for each enemy in the provided list.
        """
        noisy_distances = gameState.getAgentDistances()
        my_pos = gameState.getAgentPosition(self.index)
        if my_pos is None:
            return {}

        # Identify teammate for triangulation
        team_indices = self.getTeam(gameState)
        teammate_index = [i for i in team_indices if i != self.index][0]
        teammate_pos = gameState.getAgentPosition(teammate_index)

        width = gameState.data.layout.width
        height = gameState.data.layout.height

        enemy_distances = {}

        for enemy_idx in enemy_indices:
            # Map out the circle of possible positions from agent's perspective
            d1 = noisy_distances[enemy_idx]
            if d1 is None:
                continue
            my_possibilities = set()
            for dx in range(-d1, d1 + 1):
                dy = d1 - abs(dx)
                for sign in [1, -1]:
                    y_coord = my_pos[1] + (dy * sign)
                    x_coord = my_pos[0] + dx
                    if 0 <= x_coord < width and 0 <= y_coord < height:
                        if not gameState.hasWall(int(x_coord), int(y_coord)):
                            my_possibilities.add((int(x_coord), int(y_coord)))

            # Map out the circle from teammate's perspective
            d2 = noisy_distances[
                enemy_idx
            ]  # Note: noisy distance is same for both team agents
            teammate_possibilities = set()
            if teammate_pos:
                for dx in range(-d2, d2 + 1):
                    dy = d2 - abs(dx)
                    for sign in [1, -1]:
                        y_coord = teammate_pos[1] + (dy * sign)
                        x_coord = teammate_pos[0] + dx
                        if 0 <= x_coord < width and 0 <= y_coord < height:
                            if not gameState.hasWall(int(x_coord), int(y_coord)):
                                teammate_possibilities.add((int(x_coord), int(y_coord)))

            # Intersection
            if teammate_possibilities:
                possible_locs = my_possibilities.intersection(teammate_possibilities)
            else:
                possible_locs = my_possibilities

            if possible_locs:
                enemy_distances[enemy_idx] = min(
                    self.getMazeDistance(my_pos, loc) for loc in possible_locs
                )
            else:
                # Fallback to the noisy distance itself if no valid tiles found
                enemy_distances[enemy_idx] = d1

        return enemy_distances


####################
# Offensive Agent  #
####################


class OffensiveMCTSAgent(MCTSCaptureAgent):
    """
    Seeks food, avoids active ghosts, crosses the border, and returns
    food when carrying pellets.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Score and food count
        foodList = self.getFood(successor).asList()
        features["successorScore"] = -len(foodList)
        if foodList:
            dist = min(self.getMazeDistance(myPos, f) for f in foodList)
            features["distanceToFood"] = dist if dist else 1

        # Reward crossing into enemy territory
        mid_x = gameState.data.layout.width // 2
        x = int(myPos[0])
        features["onOffense"] = (
            1 if (self.red and x >= mid_x) or (not self.red and x < mid_x) else 0
        )

        # Incentivise returning home when carrying food
        carrying = myState.numCarrying
        if carrying > 0:
            dist = self.getMazeDistance(myPos, self.start)
            features["distanceToHome"] = dist if dist else 1
            features["carrying"] = carrying

        # Avoid active (non-scared) ghosts
        active_ghost_indices = [
            i
            for i in self.getOpponents(successor)
            if not successor.getAgentState(i).isPacman
            and not successor.getAgentState(i).scaredTimer
        ]
        scared_ghost_indices = [
            i
            for i in self.getOpponents(successor)
            if not successor.getAgentState(i).isPacman
            and successor.getAgentState(i).scaredTimer
        ]

        # Get triangulated distances for active ghosts
        active_dists = self.getEnemyPositions(gameState, active_ghost_indices)
        if active_dists:
            dist = min(active_dists.values())
            features["ghostDistance"] = dist if dist < 10 else 10
            if self.getCapsules(successor):
                dist = min(
                    self.getMazeDistance(myPos, c) for c in self.getCapsules(successor)
                )
                features["distanceToCapsule"] = dist if dist else 1
        else:
            features["ghostDistance"] = 10

        # Get triangulated distances for scared ghosts
        scared_dists = self.getEnemyPositions(gameState, scared_ghost_indices)
        if scared_dists:
            features["scaredGhostDistance"] = min(scared_dists.values())
            features["distanceToCapsule"] = (
                0  # Don't seek capsules if ghosts are already scared
            )

        # Discourage stopping and reversing
        if action == Directions.STOP:
            features["stop"] = 1
        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        if action == rev:
            features["reverse"] = 1

        return features

    # [OFFENSE_WEIGHT_START]
    def getWeights(self, gameState, action):
        score = self.getScore(gameState)
        carrying = gameState.getAgentState(self.index).numCarrying

        if score <= 0:  # losing: be more aggressive
            if carrying > 0:
                return {
                    "successorScore": 0,
                    "distanceToFood": 0,
                    "onOffense": 0,
                    "distanceToHome": -500,  # only focus on returning
                    "carrying": 0,
                    "scaredGhostDistance": 0,
                    "ghostDistance": 100,  # still tries to avoid ghosts
                    "distanceToCapsule": 0,
                    "stop": 0,
                    "reverse": 0,
                }
            else:
                return {
                    "successorScore": 1000,  # only seek food
                    "distanceToFood": -500,
                    "onOffense": 0,
                    "distanceToHome": 0,
                    "carrying": 0,
                    "scaredGhostDistance": 0,
                    "ghostDistance": 0,
                    "distanceToCapsule": 0,
                    "stop": 0,
                    "reverse": 0,
                }
        elif score >= self.foodThreshold:  # winning! be more defensive
            return {
                "successorScore": 0,
                "distanceToFood": 0,  # ignore food
                "onOffense": 0,
                "distanceToHome": -10,  # stay close to home
                "carrying": 20,  # prioritize depositing any carried food
                "scaredGhostDistance": 0,
                "ghostDistance": 10,  # still avoid ghosts
                "distanceToCapsule": 0,
                "stop": 0,
                "reverse": -2,
            }
        else:  # normal play: use tuned values from defaultCapture.lay & baselineTeam
            return {
                "successorScore": 143.12817356749218,
                "distanceToFood": -13.111191990850273,
                "onOffense": 91.73410910286486,
                "distanceToHome": -3.644067744169439,
                "carrying": 13.597119697320494,
                "scaredGhostDistance": -12.309038409442337,
                "ghostDistance": 5.4337471871544505,
                "distanceToCapsule": -4.627935674416456,
                "stop": -79.73830896027408,
                "reverse": -2.0062630339750953,
            }


# [OFFENSE_WEIGHT_END]

####################
# Defensive Agent  #
####################


class DefensiveMCTSAgent(MCTSCaptureAgent):
    """
    Keeps our side Pacman-free. Patrols the border when no invader is visible.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Prefer staying on our side of the board
        features["onDefense"] = 1
        if myState.isPacman:
            features["onDefense"] = 0

        # Chase visible invaders
        invaders = [
            i
            for i in self.getOpponents(successor)
            if successor.getAgentState(i).isPacman
            and successor.getAgentState(i).getPosition()
        ]

        features["numInvaders"] = len(invaders)
        if invaders:
            if myPos is not None:
                dists = self.getEnemyPositions(gameState, invaders)
                features["invaderDistance"] = min(dists.values()) if dists else 1
        else:
            # Patrol the border when no invader is visible
            dist = min(self.getMazeDistance(myPos, p) for p in self.patrolPoints)
            features["distanceToPatrol"] = dist if dist else 1

        # Discourage stopping and reversing
        if action == Directions.STOP:
            features["stop"] = 1
        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        if action == rev:
            features["reverse"] = 1

        return features

    # [DEFENSE_WEIGHT_START]
    def getWeights(self, gameState, action):
        score = self.getScore(gameState)
        if score >= self.foodThreshold:  # winning! be more defensive
            return {
                "numInvaders": -2000,
                "onDefense": 100,
                "invaderDistance": -200,
                "distanceToPatrol": -3,
                "stop": -100,
                "reverse": -2,
            }
        elif score <= 0:  # losing: be stable
            return {
                "numInvaders": -1000,
                "onDefense": 50,
                "invaderDistance": -200,
                "distanceToPatrol": -3,
                "stop": -100,
                "reverse": -2,
            }
        else:  # normal play: use tuned weights
            return {
                "numInvaders": -1286.0460813582645,
                "onDefense": 107.61610688600705,
                "invaderDistance": -51.56091137435831,
                "distanceToPatrol": -1.761189157964762,
                "stop": -131.92421570560703,
                "reverse": -0.5133671453882994,
            }


# [DEFENSE_WEIGHT_END]
