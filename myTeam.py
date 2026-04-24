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

import game
import util
from capture import SIGHT_RANGE, SONAR_NOISE_RANGE, GameState
from captureAgents import CaptureAgent
from game import Actions, Directions, Game

#################
# Team creation #
#################


def createTeam(
    firstIndex, secondIndex, isRed, first="DefenseAgent", second="OffenseAgent"
):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##################
# Search Problem #
##################


class SearchProblem:
    def getStartState(self):
        util.raiseNotDefined()

    def isGoalState(self, state):
        util.raiseNotDefined()

    def getSuccessors(self, state):
        util.raiseNotDefined()


def aStarSearch(problem, heuristic=lambda s, p: 0):
    """Generic A*. States are hashable (we use positions)."""
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    frontier = util.PriorityQueue()
    frontier.push((start, []), heuristic(start, problem))
    best_g = {start: 0}

    while not frontier.isEmpty():
        state, path = frontier.pop()
        if problem.isGoalState(state):
            return path
        g = best_g[state]
        for successor, action, cost in problem.getSuccessors(state):
            new_g = g + cost
            if successor not in best_g or new_g < best_g[successor]:
                best_g[successor] = new_g
                f = new_g + heuristic(successor, problem)
                frontier.push((successor, path + [action]), f)
    return []


####################
# Pathing Problems #
####################
# States here are positions (x, y). The agent chooses an action, then A* from
# the resulting position to a goal. Actions are computed from the position delta.


class PositionProblem(SearchProblem):
    """
    Base search problem for the maze. Thus goal_fn and danger_fn are
    user inputed.
    """

    def __init__(self, walls, start, goal_fn, danger_fn=None):
        self.walls = walls
        self.start = start
        self.goal_fn = goal_fn
        self.danger_fn = danger_fn or (lambda pos: 0.0)

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return self.goal_fn(state)

    def getSuccessors(self, state):
        successors = []
        x, y = state
        for action in [
            Directions.NORTH,
            Directions.SOUTH,
            Directions.EAST,
            Directions.WEST,
        ]:
            dx, dy = Actions.directionToVector(action)  # type: ignore
            nx, ny = int(x + dx), int(y + dy)
            if not self.walls[nx][ny]:
                cost = 1 + self.danger_fn((nx, ny))
                successors.append(((nx, ny), action, cost))
        return successors


###################
# Belief Tracking #
###################


class BeliefTracker:
    """
    Per-opponent belief over positions. Uses noisy manhattan-distance
    observations (SONAR_NOISE_RANGE) + exact observation when in sight.
    Transition model: uniform over legal moves (including stay, cheaply).
    """

    def __init__(self, agent, gameState):
        self.agent = agent
        self.opponents = agent.getOpponents(gameState)
        self.walls = gameState.getWalls()
        self.width = self.walls.width
        self.height = self.walls.height
        self.legal_positions = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if not self.walls[x][y]
        ]
        # beliefs[opp] = {pos: prob}
        self.beliefs = {}
        for opp in self.opponents:
            start = gameState.getInitialAgentPosition(opp)
            b = util.Counter()
            b[start] = 1.0
            self.beliefs[opp] = b

    def _noise_prior(self, true_dist):
        """P(noisy_dist | true_dist). Noise is uniform over SONAR_NOISE_RANGE values centered on 0."""
        half = (SONAR_NOISE_RANGE - 1) // 2
        return 1.0 / SONAR_NOISE_RANGE, half

    def observe(self, gameState):
        my_pos = gameState.getAgentPosition(self.agent.index)
        noisy = gameState.getAgentDistances()

        for opp in self.opponents:
            exact = gameState.getAgentPosition(opp)
            new_belief = util.Counter()

            if exact is not None:
                new_belief[exact] = 1.0
            else:
                noisy_d = noisy[opp]
                if noisy_d is None:
                    new_belief = self.beliefs[opp].copy()
                    new_belief.normalize()
                    self.beliefs[opp] = new_belief
                    continue
                p_unit, half = self._noise_prior(None)
                for pos in self.legal_positions:
                    true_d = util.manhattanDistance(my_pos, pos)
                    # If within sight range, we would have seen it; so true_d > SIGHT_RANGE
                    if true_d <= SIGHT_RANGE:
                        continue
                    # Noise is uniform in [-half, +half]
                    if abs(noisy_d - true_d) <= half:
                        new_belief[pos] = self.beliefs[opp][pos] + 1e-8

                if new_belief.totalCount() == 0:
                    # Fell off the map (shouldn't normally happen). Reset uniform.
                    for pos in self.legal_positions:
                        new_belief[pos] = 1.0

            new_belief.normalize()
            self.beliefs[opp] = new_belief

    def elapseTime(self):
        """Diffuse belief by one step of uniform random legal movement."""
        for opp in self.opponents:
            new_belief = util.Counter()
            for pos, p in self.beliefs[opp].items():
                if p == 0:
                    continue
                neighbors = Actions.getLegalNeighbors(pos, self.walls)
                # include staying put — cheap stand-in for "might not have moved"
                moves = neighbors + [pos]
                share = p / len(moves)
                for m in moves:
                    new_belief[m] += share
            new_belief.normalize()
            self.beliefs[opp] = new_belief

    def mostLikely(self, opp):
        return self.beliefs[opp].argMax()

    def danger_grid(self, gameState):
        """
        Build a [width][height] float grid: summed belief mass of opponents that
        are currently ghosts on their home side (i.e. can actually eat us).
        Scared ghosts contribute 0.
        """
        grid = [[0.0] * self.height for _ in range(self.width)]
        for opp in self.opponents:
            opp_state = gameState.getAgentState(opp)
            if opp_state.isPacman:
                continue  # they're on our side as Pacman, we're the threat
            if opp_state.scaredTimer > 1:
                continue  # scared ghost isn't dangerous
            for pos, p in self.beliefs[opp].items():
                if p > 0:
                    grid[int(pos[0])][int(pos[1])] += p
        return grid


###################
# Base Agent      #
###################


class BaseCaptureAgent(CaptureAgent):
    """Shared machinery: belief tracking, danger-aware A* helpers, home detection."""

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.walls = gameState.getWalls()
        self.width = self.walls.width
        self.height = self.walls.height
        self.mid_x = (self.width // 2) - 1 if self.red else (self.width // 2)
        self.home_boundary = [
            (self.mid_x, y) for y in range(self.height) if not self.walls[self.mid_x][y]
        ]
        self.beliefs = BeliefTracker(self, gameState)

    def observationFunction(self, gameState):
        # Must call super to get the partially-observed state.
        obs = CaptureAgent.observationFunction(self, gameState)
        return obs

    def chooseAction(self, gameState):
        self.beliefs.elapseTime()
        self.beliefs.observe(gameState)
        return self.pickAction(gameState)

    def pickAction(self, gameState):
        # We define this on our attack/defend agents
        util.raiseNotDefined()

    # --- helpers -----------------------------------------------------------

    def isHome(self, pos):
        """Is pos on our side of the map?"""
        if self.red:
            return pos[0] <= self.mid_x
        return pos[0] >= self.mid_x

    def nearestHome(self, pos):
        if not self.home_boundary:
            return pos
        return min(self.home_boundary, key=lambda b: self.getMazeDistance(pos, b))

    def buildDangerFn(self, gameState, weight=8.0):
        """Return a function pos -> danger cost, using current belief grid."""
        grid = self.beliefs.danger_grid(gameState)
        w = weight

        def danger(pos):
            x, y = int(pos[0]), int(pos[1])
            total = 0.0
            # Sum belief in a small neighborhood — being *near* a ghost is also bad
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        falloff = 1.0 if (dx == 0 and dy == 0) else 0.4
                        total += grid[nx][ny] * falloff
            return w * total

        return danger

    def planTo(self, gameState, goal_fn, danger_fn=None):
        """Run A* from current pos and return the first action, or None."""
        start = gameState.getAgentPosition(self.index)
        if start is None:
            return None
        problem = PositionProblem(self.walls, start, goal_fn, danger_fn)
        heuristic = lambda s, p: self.getMazeDistance(s, p.start) * 0  # placeholder
        # Better: distance to the *closest* goal; but we don't enumerate goals.
        # Use a cheap admissible heuristic: 0 (Dijkstra). A* with h=0 is fine for
        # the small maps we run on.
        path = aStarSearch(problem, lambda s, p: 0)
        if not path:
            return None
        return path[0]

    def pickSafestLegal(self, gameState):
        """Fallback: pick a legal action whose resulting cell has the least danger."""
        actions = gameState.getLegalActions(self.index)
        if not actions:
            return Directions.STOP
        danger_fn = self.buildDangerFn(gameState)
        my_pos = gameState.getAgentPosition(self.index)

        def score(action):
            dx, dy = Actions.directionToVector(action)
            nx, ny = int(my_pos[0] + dx), int(my_pos[1] + dy)
            d = danger_fn((nx, ny))
            # prefer not to stop
            return d + (5.0 if action == Directions.STOP else 0.0)

        return min(actions, key=score)


###################
# Offense         #
###################


class OffenseAgent(BaseCaptureAgent):
    """
    Eats food. States:
      - carrying little  -> go grab food (danger-aware)
      - carrying enough  -> run home
      - ghost adjacent   -> evade / cash in capsule if close
    """

    CARRY_LIMIT = 5  # food count that triggers return

    def pickAction(self, gameState):
        my_state = gameState.getAgentState(self.index)
        my_pos = gameState.getAgentPosition(self.index)
        carrying = my_state.numCarrying
        food_list = self.getFood(gameState).asList()
        capsules = self.getCapsules(gameState)

        # End-game sprint: only 2 food left, or time running low
        food_left = len(food_list)
        time_left = gameState.data.timeleft
        need_to_return = (
            carrying >= self.CARRY_LIMIT
            or food_left <= 2
            or time_left
            < self.getMazeDistance(my_pos, self.nearestHome(my_pos)) * 4 + 40
        )

        danger_fn = self.buildDangerFn(gameState, weight=10.0)

        # If a ghost is very close and we're pacman, try capsule first, else flee home.
        if my_state.isPacman and self._ghostAdjacent(gameState, my_pos, radius=5):
            if capsules:
                nearest_cap = min(
                    capsules, key=lambda c: self.getMazeDistance(my_pos, c)
                )
                # Only grab capsule if it's closer than home
                if self.getMazeDistance(my_pos, nearest_cap) < self.getMazeDistance(
                    my_pos, self.nearestHome(my_pos)
                ):
                    action = self.planTo(
                        gameState, lambda s: s == nearest_cap, danger_fn
                    )
                    if action:
                        return action
            # flee home
            action = self.planTo(gameState, self.isHome, danger_fn)
            if action:
                return action

        if need_to_return and carrying > 0:
            action = self.planTo(gameState, self.isHome, danger_fn)
            if action:
                return action

        # Default: grab nearest food
        if food_list:
            food_set = set(food_list)
            action = self.planTo(gameState, lambda s: s in food_set, danger_fn)
            if action:
                return action

        # Fallback
        return self.pickSafestLegal(gameState)

    def _ghostAdjacent(self, gameState, my_pos, radius=5):
        for opp in self.getOpponents(gameState):
            opp_state = gameState.getAgentState(opp)
            if opp_state.isPacman or opp_state.scaredTimer > 1:
                continue
            pos = self.beliefs.mostLikely(opp)
            if pos is None:
                continue
            if self.getMazeDistance(my_pos, pos) <= radius:
                return True
        return False


###################
# Defense         #
###################


class DefenseAgent(BaseCaptureAgent):
    """
    Patrols home side. Chases visible invaders. When no invader seen, moves
    toward the most-probable invader position from belief, or patrols boundary.
    """

    def pickAction(self, gameState):
        my_state = gameState.getAgentState(self.index)
        my_pos = gameState.getAgentPosition(self.index)

        # Identify invaders
        invaders = []
        for opp in self.getOpponents(gameState):
            opp_state = gameState.getAgentState(opp)
            if opp_state.isPacman:
                pos = opp_state.getPosition()
                if pos is None:
                    pos = self.beliefs.mostLikely(opp)
                if pos is not None:
                    invaders.append((opp, pos, opp_state))

        # Chase closest invader
        if invaders:
            target_opp, target_pos, target_state = min(
                invaders, key=lambda t: self.getMazeDistance(my_pos, t[1])
            )

            # If we're scared, keep distance (stay 2 squares away)
            if my_state.scaredTimer > 1:
                action = self._keepDistance(gameState, my_pos, target_pos, desired=2)
                if action:
                    return action
            else:
                # Don't cross into enemy territory as a ghost
                def goal(s):
                    return s == target_pos and self.isHome(s)

                # If target is on enemy side somehow, just head to boundary near them
                if not self.isHome(target_pos):
                    nearest = min(
                        self.home_boundary,
                        key=lambda b: self.getMazeDistance(b, target_pos),
                    )
                    action = self.planTo(gameState, lambda s: s == nearest)
                else:
                    action = self.planTo(
                        gameState, lambda s: s == target_pos and self.isHome(s)
                    )
                if action:
                    return action

        # No invaders: patrol. Head to the food most likely to be eaten soonest,
        # which we proxy as the food closest to the boundary.
        defending = self.getFoodYouAreDefending(gameState).asList()
        if defending:
            target = min(
                defending,
                key=lambda f: min(
                    self.getMazeDistance(f, b) for b in self.home_boundary
                ),
            )
            action = self.planTo(gameState, lambda s: s == target and self.isHome(s))
            if action:
                return action

        return self.pickSafestLegal(gameState)

    def _keepDistance(self, gameState, my_pos, target_pos, desired=2):
        """Pick the legal action that brings us to exactly `desired` maze distance."""
        actions = gameState.getLegalActions(self.index)
        best = None
        best_score = float("inf")
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            nx, ny = int(my_pos[0] + dx), int(my_pos[1] + dy)
            if not self.isHome((nx, ny)):
                continue
            d = self.getMazeDistance((nx, ny), target_pos)
            score = abs(d - desired)
            if score < best_score:
                best_score = score
                best = action
        return best
