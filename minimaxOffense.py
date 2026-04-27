from __future__ import annotations

import random
from collections import deque

from capture import SIGHT_RANGE, GameState
from captureAgents import CaptureAgent
from game import Actions, AgentState, Directions, Grid
from greedyBastards import DefenseAgent
from layout import Layout
from util import PriorityQueue, manhattanDistance

########################
# Enemy Belief Tracker #
########################


class EnemyBeliefTracker:
    """
    Tracks the belief distribution of enemy positions based on sonar noise.
    """

    def __init__(
        self,
        ally_index: int,
        enemy_index: int,
        initial_gamestate: GameState,
        naive: bool,
    ):
        self.ally_index: int = ally_index
        self.enemy_index: int = enemy_index
        self.initial_enemy_position = initial_gamestate.getInitialAgentPosition(
            enemy_index
        )
        self.naive: bool = naive
        self.enemy_position_distribution: set = {self.initial_enemy_position}
        self.layout: Layout = initial_gamestate.data.layout
        self.max_x: int = self.layout.width - 1
        self.max_y: int = self.layout.height - 1

    def update_naive(self, gameState: GameState):
        """
        Update the belief distribution based on the new given game state
        Method is as such:
            1. stretch the distribution to up, down, left, right by 1 (essentially disable
                current cell, add one to each of up, down, left, right of such cell (only
                legal ones though, essentially flooding enemy cells.))
            2. Define the square defining the possible position of the cell given
                the by the enemyDistance (with noise), then intersect the square with
                the stretched distribution
        """

        # 1. Exact observation
        enemy_pos = gameState.getAgentPosition(self.enemy_index)
        if enemy_pos is not None:
            self.enemy_position_distribution = {enemy_pos}
            return

        # Diffuse prior by one step (enemy may have moved)
        stretched: set = set()
        for x, y in self.enemy_position_distribution:
            for nx, ny in [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if nx < 0 or nx > self.max_x or ny < 0 or ny > self.max_y:
                    continue
                if self.layout.isWall((nx, ny)):
                    continue
                stretched.add((nx, ny))

        # 2. Noisy observation via state.noise
        noise_list = gameState.noise
        noise_config = noise_list[self.enemy_index] if noise_list else None
        if noise_config is not None and noise_config.pos is not None:
            nx_obs, ny_obs = noise_config.pos
            nx_obs, ny_obs = int(nx_obs), int(ny_obs)
            # True position lies in [obs - 2, obs + 2] in each axis
            box = set()
            for x in range(nx_obs - 2, nx_obs + 3):
                for y in range(ny_obs - 2, ny_obs + 3):
                    if x < 0 or x > self.max_x or y < 0 or y > self.max_y:
                        continue
                    if self.layout.isWall((x, y)):
                        continue
                    box.add((x, y))

            new_belief = stretched & box
            # Also exclude positions within sight range — we'd have seen them
            ally_pos = gameState.getAgentPosition(self.ally_index)
            if ally_pos is not None:
                new_belief = {
                    p
                    for p in new_belief
                    if manhattanDistance(ally_pos, p) > SIGHT_RANGE
                }

            # Fallback: if intersection killed everything, trust the noise box alone
            if not new_belief:
                new_belief = box
        else:
            # 3. No signal — just diffuse
            new_belief = stretched

        if not new_belief:
            new_belief = {self.initial_enemy_position}

        self.enemy_position_distribution = new_belief

    def update_non_naive(self, gameState: GameState):
        self.update_naive(gameState)

    def update_with_ally(self, ally: EnemyBeliefTracker):
        """
        Intersect our enemy position belief with ally enemy position belief
        Only need to call this once per pair, as this syncs to the other agent also.
        """
        new_belief = self.enemy_position_distribution.intersection(
            ally.enemy_position_distribution
        )
        if new_belief:
            self.enemy_position_distribution = new_belief
            ally.enemy_position_distribution = new_belief

    def get_proximity_value(self, pos):
        """
        Gets proximity value from position to current.
        """

        if not self.enemy_position_distribution:
            return float("inf")
        else:
            return sum(
                manhattanDistance(position, pos)
                for position in self.enemy_position_distribution
            ) / len(self.enemy_position_distribution)

    def closest_distance(self, pos):
        """Min manhattan distance from pos to any believed enemy position."""
        if not self.enemy_position_distribution:
            return float("inf")
        return min(
            manhattanDistance(position, pos)
            for position in self.enemy_position_distribution
        )

    def update(self, gameState):
        if self.naive:
            self.update_naive(gameState)
        else:
            self.update_non_naive(gameState)


###################
# Offensive Agent #
###################


class GreedyPointAgent(CaptureAgent):
    """
    Agent idea (this is offense agent btw):

    If there are no enemies in the proximity:
        use A* to greedy food capture (heavily penalizes going to foods in tunnels and
        blockades, as we can be tricked by enemies to go to a food that is safe for us.)
    Else:
        If we have more food than min_threshold:
            minimax home
        If there is capsule close:
            minimax for it
            if minimax succeeds:
                use it
            else:
                minimax home
        Else:
            If there is food close:
                check if it's safe to eat it (we construct
                a path from enemy to us, then construct a path from us to the food,
                and check if the paths is clear.
                if safe:
                    eat it
                else:
                    rush home
            If not:
                rush home

    """

    # ------ Tunables ------
    CARRY_THRESHOLD = 5  # carry this many before considering home
    DANGER_RADIUS = 4  # within this maze distance an enemy ghost is "near"
    MINIMAX_DEPTH = 4  # plies (each ply = one of our + one ghost move)
    SAFE_FOOD_RADIUS = 6  # food considered "close" for safety check
    CAPSULE_RUSH_RADIUS = 7  # capsule considered "close" enough to rush

    def registerInitialState(self, gameState: GameState):
        CaptureAgent.registerInitialState(self, gameState)

        # Get the walls for traversal
        self.walls = gameState.getWalls()
        self.width = self.walls.width
        self.height = self.walls.height

        # Get the food for food capture
        my_food = self.getFoodYouAreDefending(gameState)
        self.my_food_set = set(my_food.asList())

        # Get our food to predict enemy location
        op_food = self.getFood(gameState)
        self.enemy_food_set = set(op_food.asList())

        # Get the capsules for capsule capture
        self.capsules = gameState.getCapsules()

        # Get midpoint for determining ours versus enemy's
        self.x_mid = self.walls.width // 2

        # Keep enemy indices just cause.
        self.enemy_indices = self.getOpponents(gameState)

        # Setup initial distribution for the enemy distances as
        # P(pos) = 1 if enemy intial position else 0
        self.enemy_distance_distribution = {
            enemy: set() for enemy in self.enemy_indices
        }
        for enemy in self.enemy_indices:
            self.enemy_distance_distribution[enemy].add(
                gameState.getInitialAgentPosition(enemy)
            )

        # Belief trackers — one per enemy (single ally view, since cross-agent
        # ally syncing would require shared state across agent instances)
        self.belief_trackers = {
            enemy: EnemyBeliefTracker(self.index, enemy, gameState, naive=True)
            for enemy in self.enemy_indices
        }

        # Home boundary cells (our side, on the dividing column)
        self.start = gameState.getAgentPosition(self.index)
        self.home_x = (self.width // 2) - 1 if self.red else (self.width // 2)
        self.home_boundary = [
            (self.home_x, y)
            for y in range(self.height)
            if not self.walls[self.home_x][y]
        ]

    # -------------------- main entry --------------------

    def chooseAction(self, gameState: GameState):
        # Update beliefs from latest observation
        for tracker in self.belief_trackers.values():
            tracker.update(gameState)

        my_pos = gameState.getAgentPosition(self.index)
        my_state = gameState.getAgentState(self.index)
        carrying = my_state.numCarrying
        food_list = self.getFood(gameState).asList()
        capsules = self.getCapsules(gameState)
        food_left = len(food_list)
        time_left = gameState.data.timeleft

        # Threat assessment: nearest dangerous (non-scared, ghost-form) enemy
        threat_dist, threat_pos, threat_idx = self._nearestThreat(gameState, my_pos)
        in_danger = (
            my_state.isPacman
            and threat_dist is not None
            and threat_dist <= self.DANGER_RADIUS
        )

        # Endgame: must run home
        home_dist = self._distToHome(my_pos)
        must_return = (
            (carrying >= self.CARRY_THRESHOLD)
            or (food_left <= 2 and carrying > 0)
            or (carrying > 0 and time_left < home_dist * 4 + 40)
        )

        # --- Branch 1: enemy nearby ---
        if in_danger:
            # Already carrying enough -> minimax home
            if carrying >= self.CARRY_THRESHOLD or must_return:
                print("Carry enough, in danger, return to home")
                action = self._minimaxEscape(
                    gameState, prefer="home", threat_idx=threat_idx
                )
                if action is not None:
                    return action

            # Capsule close
            if capsules:
                print("Capsule close, in danger, rush capsule")
                nearest_cap = min(
                    capsules, key=lambda c: self.getMazeDistance(my_pos, c)
                )
                cap_dist = self.getMazeDistance(my_pos, nearest_cap)
                if cap_dist <= self.CAPSULE_RUSH_RADIUS and cap_dist <= home_dist:
                    action = self._minimaxEscape(
                        gameState,
                        prefer="capsule",
                        threat_idx=threat_idx,
                        capsule=nearest_cap,
                    )
                    if action is not None:
                        return action

            # get close food
            if food_list:
                print("Safe food close, in danger, eat safe food")
                close_food = [
                    f
                    for f in food_list
                    if self.getMazeDistance(my_pos, f) <= self.SAFE_FOOD_RADIUS
                ]
                if close_food:
                    safe = self._safestFood(gameState, my_pos, close_food, threat_pos)
                    if safe is not None:
                        action = self._aStarFirstAction(
                            gameState,
                            my_pos,
                            lambda s: s == safe,
                            danger_fn=self._buildDangerFn(gameState),
                        )
                        if action is not None:
                            return action

            # Default: rush home with minimax
            print("In danger, rush home")
            action = self._minimaxEscape(
                gameState, prefer="home", threat_idx=threat_idx
            )
            if action is not None:
                return action

        # --- Branch 2: no immediate threat ---
        if must_return and carrying > 0:
            print("No danger, carry too much, return to home")
            action = self._aStarFirstAction(
                gameState,
                my_pos,
                self._isHome,
                danger_fn=self._buildDangerFn(gameState),
            )
            if action is not None:
                return action

        # Greedy food capture with light danger awareness
        if food_list:
            print("Greedy food get")
            food_set = set(food_list)
            action = self._aStarFirstAction(
                gameState,
                my_pos,
                lambda s: s in food_set,
                danger_fn=self._buildDangerFn(gameState, weight=4.0),
            )
            if action is not None:
                return action

        # Fallback: any safe legal move
        return self._safestLegal(gameState, my_pos)

    # -------------------- threat helpers --------------------

    def _nearestThreat(self, gameState, my_pos):
        """Return (maze_dist, pos, enemy_idx) of nearest dangerous ghost, or (None, None, None)."""
        best = (None, None, None)
        for enemy in self.enemy_indices:
            enemy_state: AgentState = gameState.getAgentState(enemy)
            # On their home side as ghost, not scared = threat
            if enemy_state.isPacman:
                continue
            if enemy_state.scaredTimer > 1:
                continue
            exact = gameState.getAgentPosition(enemy)
            if exact is None:
                # Use closest point in belief if it's plausibly close
                belief = self.belief_trackers[enemy].enemy_position_distribution
                if not belief:
                    continue
                pos = min(belief, key=lambda p: self.getMazeDistance(my_pos, p))
            else:
                pos = exact
            try:
                d = self.getMazeDistance(my_pos, pos)
            except Exception:
                d = manhattanDistance(my_pos, pos)
            if best[0] is None or d < best[0]:
                best = (d, pos, enemy)
        return best

    def _buildDangerFn(self, gameState, weight=10.0):
        """Cost overlay: positions adjacent to threatening ghosts cost more."""
        threats = []
        for enemy in self.enemy_indices:
            es = gameState.getAgentState(enemy)
            if es.isPacman or es.scaredTimer > 1:
                continue
            exact = gameState.getAgentPosition(enemy)
            if exact is not None:
                threats.append(exact)
            else:
                belief = self.belief_trackers[enemy].enemy_position_distribution
                # Use up to a few high-weight points
                threats.extend(list(belief)[:8])
        if not threats:
            return lambda pos: 0.0

        def danger(pos):
            total = 0.0
            for t in threats:
                d = manhattanDistance(pos, t)
                if d == 0:
                    total += weight * 5
                elif d == 1:
                    total += weight * 2
                elif d == 2:
                    total += weight * 0.6
                elif d == 3:
                    total += weight * 0.2
            return total

        return danger

    # -------------------- A* pathing --------------------

    def _aStarFirstAction(self, gameState, start, goal_fn, danger_fn=None):
        """A* on the maze grid; return first action of optimal path or None."""
        if danger_fn is None:
            danger_fn = lambda p: 0.0
        if goal_fn(start):
            return Directions.STOP

        frontier = PriorityQueue()
        frontier.push((start, []), 0)
        best_g = {start: 0.0}

        while not frontier.isEmpty():
            state, path = frontier.pop()
            if goal_fn(state):
                return path[0] if path else Directions.STOP
            g = best_g[state]
            x, y = state
            for action in (
                Directions.NORTH,
                Directions.SOUTH,
                Directions.EAST,
                Directions.WEST,
            ):
                dx, dy = Actions.directionToVector(action)
                nx, ny = int(x + dx), int(y + dy)
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                if self.walls[nx][ny]:
                    continue
                step_cost = 1.0 + danger_fn((nx, ny))
                new_g = g + step_cost
                succ = (nx, ny)
                if succ not in best_g or new_g < best_g[succ]:
                    best_g[succ] = new_g
                    # heuristic: 0 (Dijkstra) — admissible & cheap
                    frontier.push((succ, path + [action]), new_g)
        return None

    # -------------------- minimax escape --------------------

    def _minimaxEscape(self, gameState, prefer, threat_idx, capsule=None):
        """
        Lightweight minimax: we maximize, threat ghost minimizes.
        Search depth = MINIMAX_DEPTH plies. Evaluate leaf states by a heuristic
        balancing distance to goal (home or capsule) against distance to ghost.
        """
        my_pos = gameState.getAgentPosition(self.index)
        if threat_idx is None:
            return None

        # If ghost is unobservable, we still need a pos to simulate against —
        # use the closest belief point as a proxy.
        ghost_pos = gameState.getAgentPosition(threat_idx)
        if ghost_pos is None:
            belief = self.belief_trackers[threat_idx].enemy_position_distribution
            if not belief:
                return None
            ghost_pos = min(belief, key=lambda p: self.getMazeDistance(my_pos, p))
            ghost_pos = (int(ghost_pos[0]), int(ghost_pos[1]))

        def eval_state(me, ghost, depth_left):
            # Distance to goal
            if prefer == "capsule" and capsule is not None:
                try:
                    goal_d = self.getMazeDistance(me, capsule)
                except Exception:
                    goal_d = manhattanDistance(me, capsule)
            else:
                goal_d = self._distToHome(me)
            try:
                ghost_d = self.getMazeDistance(me, ghost)
            except Exception:
                ghost_d = manhattanDistance(me, ghost)
            # Caught: very bad
            if ghost_d == 0:
                return -1e6
            # Reward: short goal distance, long ghost distance
            score = -goal_d * 10 + min(ghost_d, 6) * 4
            # Bonus for already being safe (home side)
            if self._isHome(me):
                score += 50
            if prefer == "capsule" and capsule is not None and me == capsule:
                score += 200
            return score

        def neighbors(pos):
            x, y = pos
            out = [pos]  # allow staying
            for action in (
                Directions.NORTH,
                Directions.SOUTH,
                Directions.EAST,
                Directions.WEST,
            ):
                dx, dy = Actions.directionToVector(action)
                nx, ny = int(x + dx), int(y + dy)
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and not self.walls[nx][ny]
                ):
                    out.append(((nx, ny), action))
            # First entry is "stay" without action; reformat
            return out

        def my_moves(pos):
            x, y = pos
            results = []
            for action in (
                Directions.NORTH,
                Directions.SOUTH,
                Directions.EAST,
                Directions.WEST,
                Directions.STOP,
            ):
                dx, dy = Actions.directionToVector(action)
                nx, ny = int(x + dx), int(y + dy)
                if action == Directions.STOP:
                    results.append((pos, action))
                    continue
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and not self.walls[nx][ny]
                ):
                    results.append(((nx, ny), action))
            return results

        def ghost_moves(pos):
            x, y = pos
            results = []
            for action in (
                Directions.NORTH,
                Directions.SOUTH,
                Directions.EAST,
                Directions.WEST,
            ):
                dx, dy = Actions.directionToVector(action)
                nx, ny = int(x + dx), int(y + dy)
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and not self.walls[nx][ny]
                ):
                    results.append((nx, ny))
            if not results:
                results.append(pos)
            return results

        def maxnode(me, ghost, depth, alpha, beta):
            if depth == 0 or me == ghost:
                return eval_state(me, ghost, depth), None
            best_v = float("-inf")
            best_a = None
            for new_me, action in my_moves(me):
                v, _ = minnode(new_me, ghost, depth - 1, alpha, beta)
                if v > best_v:
                    best_v = v
                    best_a = action
                alpha = max(alpha, best_v)
                if beta <= alpha:
                    break
            return best_v, best_a

        def minnode(me, ghost, depth, alpha, beta):
            if depth == 0 or me == ghost:
                return eval_state(me, ghost, depth), None
            best_v = float("inf")
            for new_ghost in ghost_moves(ghost):
                v, _ = maxnode(me, new_ghost, depth - 1, alpha, beta)
                if v < best_v:
                    best_v = v
                alpha_local = alpha
                beta = min(beta, best_v)
                if beta <= alpha_local:
                    break
            return best_v, None

        _, action = maxnode(
            my_pos, ghost_pos, self.MINIMAX_DEPTH * 2, float("-inf"), float("inf")
        )
        if action is None:
            return None
        # Verify the action is legal in the actual game state
        legal = gameState.getLegalActions(self.index)
        if action not in legal:
            return None
        return action

    # -------------------- safe food check --------------------

    def _safestFood(self, gameState, my_pos, foods, threat_pos):
        """
        Return the food whose path doesn't get intercepted: distance from us
        to food is meaningfully shorter than ghost distance to food.
        """
        if threat_pos is None:
            return min(foods, key=lambda f: self.getMazeDistance(my_pos, f))

        best = None
        best_score = float("-inf")
        for f in foods:
            try:
                my_d = self.getMazeDistance(my_pos, f)
                ghost_d = self.getMazeDistance(threat_pos, f)
            except Exception:
                my_d = manhattanDistance(my_pos, f)
                ghost_d = manhattanDistance(threat_pos, f)
            # We want ghost_d - my_d to be large (we get there well before)
            margin = ghost_d - my_d
            if margin >= 2:  # need a real cushion
                score = margin - my_d * 0.1
                if score > best_score:
                    best_score = score
                    best = f
        return best

    # -------------------- side helpers --------------------

    def _isHome(self, pos):
        x = int(pos[0])
        if self.red:
            return x <= self.home_x
        return x >= self.home_x

    def _distToHome(self, pos):
        if self._isHome(pos):
            return 0
        if not self.home_boundary:
            return 0
        return min(self.getMazeDistance(pos, b) for b in self.home_boundary)

    def _safestLegal(self, gameState, my_pos):
        actions = gameState.getLegalActions(self.index)
        if not actions:
            return Directions.STOP
        danger_fn = self._buildDangerFn(gameState)

        def score(action):
            dx, dy = Actions.directionToVector(action)
            nx, ny = int(my_pos[0] + dx), int(my_pos[1] + dy)
            d = danger_fn((nx, ny))
            return d + (5.0 if action == Directions.STOP else 0.0)

        return min(actions, key=score)


def createTeam(
    firstIndex, secondIndex, isRed, first="GreedyPointAgent", second="DefenseAgent"
):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]
