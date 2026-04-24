# myTeam3.py
# ----------
# Strategic agent with offline map analysis, pre-planned strategy tree,
# short-horizon minimax, and score-aware play.
#
# ============================================================================
# STRATEGY TREE (evaluated top-down each turn per agent)
# ============================================================================
#
# ROOT
# ├── Score-aware meta-mode
# │   ├── WINNING_BIG (score >= +6)       -> both agents DEFEND
# │   ├── LOSING_BIG  (score <= -6)       -> both agents ATTACK
# │   ├── END_GAME    (time < 80 moves)
# │   │   ├── if carrying food            -> RUSH_HOME
# │   │   ├── if we're ahead              -> DEFEND
# │   │   └── else                        -> ATTACK
# │   └── DEFAULT                         -> role-based (one offense, one defense)
# │
# ├── OFFENSE role
# │   ├── THREAT present (ghost can catch us before home)
# │   │   ├── capsule closer than home    -> GRAB_CAPSULE
# │   │   └── else                        -> RETREAT_HOME (avoid dead-ends)
# │   ├── IN_TUNNEL and ghost near mouth  -> RETREAT_HOME
# │   ├── carrying >= CARRY_LIMIT         -> RETREAT_HOME
# │   ├── food_left <= 2                  -> RUSH_HOME
# │   ├── SCARED_TIME active              -> AGGRESSIVE_EAT (minimax vs feast)
# │   └── default                         -> EAT_FOOD (minimax vs nearest ghost)
# │
# └── DEFENSE role
#     ├── VISIBLE_INVADER                 -> CHASE (minimax vs invader)
#     │   └── if scared                   -> STALL (distance=2)
#     ├── food-just-eaten                 -> INVESTIGATE (head to eaten cell)
#     ├── believed invader near choke     -> BLOCK_CHOKE
#     └── default                         -> PATROL_CHOKES (rotate through choke points)
#
# ============================================================================
# OFFLINE MAP ANALYSIS (done once in registerInitialState)
# ============================================================================
#   - Dead-end tunnels: cells with only one non-wall neighbor, propagated
#     through 2-neighbor cells. Each tunnel cell stores its depth.
#   - Choke points: articulation points in the home-side graph. These are
#     cells whose removal disconnects defended food from the boundary.
#   - Boundary crossings: cells on the midline with no wall (entry/exit gates).
#
# ============================================================================
# MINIMAX
# ============================================================================
#   Alpha-beta over (us, nearest_relevant_opponent) at depth 2-4 (adaptive).
#   Other opponents held fixed at belief mode. Eval = weighted feature sum.
# ============================================================================

import random

import util
from capture import SIGHT_RANGE, GameState
from captureAgents import CaptureAgent
from game import Actions, Directions

#################
# Team creation #
#################


def createTeam(
    firstIndex, secondIndex, isRed, first="StrategicAgent", second="StrategicAgent"
):
    return [
        eval(first)(firstIndex, role="offense"),
        eval(second)(secondIndex, role="defense"),
    ]


#############################
# Offline map analysis      #
#############################


class MapAnalysis:
    """Precomputed structural facts about the layout. One instance per game."""

    def __init__(self, gameState, agent):
        self.walls = gameState.getWalls()
        self.width = self.walls.width
        self.height = self.walls.height
        self.red = agent.red
        self.mid_x = (self.width // 2) - 1 if self.red else (self.width // 2)
        self.enemy_mid_x = self.width // 2 if self.red else (self.width // 2) - 1

        self.legal = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if not self.walls[x][y]
        ]
        self.legal_set = set(self.legal)

        self.home_cells = {p for p in self.legal if self._isHome(p)}
        self.enemy_cells = self.legal_set - self.home_cells

        # Boundary = home cells adjacent to enemy cells
        self.boundary = [
            p
            for p in self.home_cells
            if any(
                (p[0] + dx, p[1] + dy) in self.enemy_cells
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
        ]

        # Dead-end / tunnel depths: dict pos -> depth (0 = not a tunnel)
        self.tunnel_depth = self._computeTunnelDepths()

        # Choke points on home side
        self.choke_points = self._computeChokePoints()

    def _isHome(self, pos):
        if self.red:
            return pos[0] <= self.mid_x
        return pos[0] >= self.mid_x

    def _neighbors(self, pos):
        x, y = pos
        result = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < self.width
                and 0 <= ny < self.height
                and not self.walls[nx][ny]
            ):
                result.append((nx, ny))
        return result

    def _computeTunnelDepths(self):
        """
        A tunnel cell is one from which the only way out passes through
        another tunnel cell or a dead end. Depth = how deep into the tunnel.
        Algorithm: iteratively mark cells with <= 1 non-tunnel neighbor.
        """
        depth = {p: 0 for p in self.legal}
        current_depth = 1
        changed = True
        while changed:
            changed = False
            for p in self.legal:
                if depth[p] > 0:
                    continue
                open_neighbors = [n for n in self._neighbors(p) if depth[n] == 0]
                if len(open_neighbors) <= 1:
                    depth[p] = current_depth
                    changed = True
            current_depth += 1
            if current_depth > 20:  # safety
                break
        return depth

    def _computeChokePoints(self):
        """
        Articulation points in the home-side subgraph. Naive O(V*(V+E)) version:
        for each non-boundary home cell, remove it, check if boundary cells can
        still reach all defended-food candidate cells. Since the graph is small
        this is fine (run once at startup, <15 seconds).
        """
        chokes = []
        home_graph = self.home_cells
        if not self.boundary:
            return chokes

        # Interior home cells (not on boundary) that could be articulation points
        interior = [p for p in home_graph if p not in set(self.boundary)]

        def reachable_from(start, forbidden, graph):
            seen = {start}
            stack = [start]
            while stack:
                cur = stack.pop()
                for n in self._neighbors(cur):
                    if n in graph and n not in forbidden and n not in seen:
                        seen.add(n)
                        stack.append(n)
            return seen

        full_reach = reachable_from(self.boundary[0], set(), home_graph)

        for candidate in interior:
            # Remove candidate, see if anything becomes unreachable from boundary
            reach = reachable_from(self.boundary[0], {candidate}, home_graph)
            if len(reach) < len(full_reach) - 1:  # -1 because candidate itself is gone
                chokes.append(candidate)

        # If too few chokes found, fall back to boundary cells themselves
        if len(chokes) < 2:
            chokes = list(self.boundary)
        return chokes

    def tunnelMouth(self, pos):
        """If pos is in a tunnel, return the cell just outside the tunnel mouth."""
        if self.tunnel_depth.get(pos, 0) == 0:
            return None
        # Walk toward shallower depth until we exit
        current = pos
        seen = {current}
        while self.tunnel_depth[current] > 0:
            neighbors = [n for n in self._neighbors(current) if n not in seen]
            if not neighbors:
                return current
            # Move toward lower depth (out of tunnel)
            neighbors.sort(key=lambda n: self.tunnel_depth[n])
            current = neighbors[0]
            seen.add(current)
            if len(seen) > 30:
                break
        return current


#############################
# Belief tracking (compact) #
#############################


class SimpleBelief:
    """Belief tracker. Kept separate from the full HMM — simpler and faster."""

    def __init__(self, agent, gameState, map_analysis):
        self.agent = agent
        self.map = map_analysis
        self.opponents = agent.getOpponents(gameState)
        self.beliefs = {}
        for opp in self.opponents:
            b = util.Counter()
            b[gameState.getInitialAgentPosition(opp)] = 1.0
            self.beliefs[opp] = b
        self.prev_defended = set(agent.getFoodYouAreDefending(gameState).asList())
        self.recently_eaten = set()

    def update(self, gameState):
        # Diffuse
        for opp in self.opponents:
            new_b = util.Counter()
            for pos, p in self.beliefs[opp].items():
                if p == 0:
                    continue
                moves = [pos] + [n for n in self.map._neighbors(pos)]
                share = p / len(moves)
                for m in moves:
                    new_b[m] += share
            self.beliefs[opp] = new_b

        # Observe: sight + range
        my_pos = gameState.getAgentPosition(self.agent.index)
        for opp in self.opponents:
            exact = gameState.getAgentPosition(opp)
            if exact is not None:
                b = util.Counter()
                b[exact] = 1.0
                self.beliefs[opp] = b
            else:
                # Zero out cells within sight range (we would've seen them)
                for pos in list(self.beliefs[opp].keys()):
                    if util.manhattanDistance(my_pos, pos) <= SIGHT_RANGE:
                        self.beliefs[opp][pos] = 0
                self.beliefs[opp].normalize()
                if self.beliefs[opp].totalCount() == 0:
                    for pos in self.map.legal:
                        if util.manhattanDistance(my_pos, pos) > SIGHT_RANGE:
                            self.beliefs[opp][pos] = 1.0
                    self.beliefs[opp].normalize()

        # Food events
        current = set(self.agent.getFoodYouAreDefending(gameState).asList())
        self.recently_eaten = self.prev_defended - current
        self.prev_defended = current
        for pos in self.recently_eaten:
            best_opp = max(self.opponents, key=lambda o: self.beliefs[o][pos])
            b = util.Counter()
            b[pos] = 1.0
            self.beliefs[best_opp] = b

    def mostLikely(self, opp):
        return self.beliefs[opp].argMax()


#############################
# Pathing                   #
#############################


def astar(walls, start, goal_fn, cost_fn=None):
    """A* with optional danger cost. Returns list of actions."""
    if goal_fn(start):
        return []
    if cost_fn is None:
        cost_fn = lambda p: 1.0

    frontier = util.PriorityQueue()
    frontier.push((start, []), 0)
    best_g = {start: 0}

    while not frontier.isEmpty():
        state, path = frontier.pop()
        if goal_fn(state):
            return path
        x, y = state
        for action in [
            Directions.NORTH,
            Directions.SOUTH,
            Directions.EAST,
            Directions.WEST,
        ]:
            dx, dy = Actions.directionToVector(action)
            nx, ny = int(x + dx), int(y + dy)
            if walls[nx][ny]:
                continue
            new_g = best_g[state] + cost_fn((nx, ny))
            if (nx, ny) not in best_g or new_g < best_g[(nx, ny)]:
                best_g[(nx, ny)] = new_g
                frontier.push(((nx, ny), path + [action]), new_g)
    return []


#############################
# Main agent                #
#############################

# Weight vectors used by the evaluation function. Separated by role so each
# agent optimizes the features it actually cares about.
OFFENSE_WEIGHTS = {
    "score": 100.0,
    "food_left": -3.0,
    "carrying": 15.0,  # (times carrying ^ 0.5 to diminish)
    "dist_to_food": -1.0,
    "dist_to_home": -0.5,  # only when carrying > 0
    "ghost_penalty": -500.0,  # within 2 steps
    "ghost_proximity": -20.0,
    "capsule_bonus": 5.0,
    "in_dead_end": -30.0,  # tunnel depth when ghost near
    "scared_bonus": 20.0,
}

DEFENSE_WEIGHTS = {
    "score": 50.0,
    "num_invaders": -200.0,
    "dist_to_invader": -8.0,
    "on_defense": 50.0,  # negative if we became pacman
    "dist_to_choke": -1.0,
    "dist_to_eaten": -5.0,  # toward recently-eaten cell
    "stop_penalty": -10.0,
    "reverse_penalty": -2.0,
    "defended_food": 2.0,
}


class StrategicAgent(CaptureAgent):
    CARRY_LIMIT = 5
    END_GAME_MOVES = 80  # < this many total moves left = end game
    WINNING_BIG = 6
    LOSING_BIG = -6

    # Shared across both agents on the team (class-level dict keyed by game id)
    _team_state = {}

    def __init__(self, index, role="offense", timeForComputing=0.1):
        super().__init__(index, timeForComputing)
        self.role = role

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.map = MapAnalysis(gameState, self)
        self.beliefs = SimpleBelief(self, gameState, self.map)
        self.patrol_idx = 0
        self.last_action = Directions.STOP
        # Pick weights for our role
        self.weights = OFFENSE_WEIGHTS if self.role == "offense" else DEFENSE_WEIGHTS

    # -----------------------------------------------------------------------
    # Meta-mode selection (top of strategy tree)
    # -----------------------------------------------------------------------

    def chooseAction(self, gameState):
        self.beliefs.update(gameState)
        mode = self._decideMode(gameState)
        my_pos = gameState.getAgentPosition(self.index)
        my_state = gameState.getAgentState(self.index)

        # Dispatch on mode
        if mode == "rush_home":
            return self._planAction(gameState, self._goalHome, danger=True)
        if mode == "defend":
            return self._defenseAction(gameState)
        if mode == "attack":
            return self._offenseAction(gameState)
        if mode == "role":
            if self.role == "offense":
                return self._offenseAction(gameState)
            else:
                return self._defenseAction(gameState)
        return self._safestLegal(gameState)

    def _decideMode(self, gameState):
        score = self.getScore(gameState)
        time_left = gameState.data.timeleft
        my_state = gameState.getAgentState(self.index)

        # End-game: commit based on carrying & lead
        if time_left < self.END_GAME_MOVES:
            if my_state.numCarrying > 0:
                return "rush_home"
            if score > 0:
                return "defend"
            return "attack"

        # Score-aware overrides
        if score >= self.WINNING_BIG:
            return "defend"
        if score <= self.LOSING_BIG:
            return "attack"

        return "role"

    # -----------------------------------------------------------------------
    # Offense
    # -----------------------------------------------------------------------

    def _offenseAction(self, gameState):
        my_pos = gameState.getAgentPosition(self.index)
        my_state = gameState.getAgentState(self.index)
        food = self.getFood(gameState).asList()
        capsules = self.getCapsules(gameState)

        # Check for active scared-ghost time: if any enemy is scared with time left,
        # we can feast with less caution.
        scared_active = any(
            gameState.getAgentState(o).scaredTimer > 3
            for o in self.getOpponents(gameState)
        )

        carrying = my_state.numCarrying
        threat = self._realThreat(gameState, my_pos, my_state)

        # Tunnel trap: if we're deep in a tunnel and ghost is near the mouth, retreat
        if my_state.isPacman and self.map.tunnel_depth.get(my_pos, 0) > 0:
            mouth = self.map.tunnelMouth(my_pos)
            if mouth and self._ghostNear(
                gameState, mouth, radius=self.map.tunnel_depth[my_pos] + 1
            ):
                return self._planAction(gameState, self._goalHome, danger=True)

        # Hard threat -> capsule if affordable, else home
        if threat and my_state.isPacman and not scared_active:
            if capsules:
                home_d = self._homeDist(my_pos)
                nearest_cap = min(
                    capsules, key=lambda c: self.getMazeDistance(my_pos, c)
                )
                if self.getMazeDistance(my_pos, nearest_cap) < home_d:
                    return self._planAction(
                        gameState, lambda s: s == nearest_cap, danger=True
                    )
            return self._planAction(gameState, self._goalHome, danger=True)

        # Quota / end-game retreat
        if carrying >= self.CARRY_LIMIT or len(food) <= 2:
            return self._planAction(gameState, self._goalHome, danger=True)

        # Otherwise: minimax to decide the actual move
        return self._minimaxAction(gameState)

    # -----------------------------------------------------------------------
    # Defense
    # -----------------------------------------------------------------------

    def _defenseAction(self, gameState):
        my_pos = gameState.getAgentPosition(self.index)
        my_state = gameState.getAgentState(self.index)

        invaders = self._visibleInvaders(gameState)

        if invaders:
            _, target_pos, target_state = min(
                invaders, key=lambda t: self.getMazeDistance(my_pos, t[1])
            )
            if my_state.scaredTimer > 1:
                return self._keepDistance(gameState, my_pos, target_pos, desired=2)
            # Use minimax to chase — handles the invader dodging us
            return self._minimaxAction(gameState)

        # No visible invader: head to recently-eaten cell
        if self.beliefs.recently_eaten:
            target = min(
                self.beliefs.recently_eaten,
                key=lambda p: self.getMazeDistance(my_pos, p),
            )
            return self._planAction(
                gameState, lambda s: s == target and s in self.map.home_cells
            )

        # Check if belief concentrates near a choke point
        for opp in self.getOpponents(gameState):
            opp_state = gameState.getAgentState(opp)
            if not opp_state.isPacman:
                continue
            ml = self.beliefs.mostLikely(opp)
            if ml and ml in self.map.home_cells:
                # Go block the closest choke to believed invader
                if self.map.choke_points:
                    choke = min(
                        self.map.choke_points, key=lambda c: self.getMazeDistance(c, ml)
                    )
                    return self._planAction(gameState, lambda s: s == choke)

        # Default: patrol choke points
        if self.map.choke_points:
            target = self.map.choke_points[self.patrol_idx % len(self.map.choke_points)]
            if self.getMazeDistance(my_pos, target) <= 1:
                self.patrol_idx = (self.patrol_idx + 1) % len(self.map.choke_points)
                target = self.map.choke_points[self.patrol_idx]
            return self._planAction(gameState, lambda s: s == target)

        return self._safestLegal(gameState)

    # -----------------------------------------------------------------------
    # Minimax with alpha-beta
    # -----------------------------------------------------------------------

    def _minimaxAction(self, gameState):
        """Find the best action at depth 2-4 against the nearest relevant opponent."""
        opp_index = self._relevantOpponent(gameState)

        depth = 3
        # Shallower search if multiple opponents close
        nearby = sum(
            1
            for o in self.getOpponents(gameState)
            if self._distToOpp(gameState, o) <= 5
        )
        if nearby >= 2:
            depth = 2

        best_action = Directions.STOP
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in gameState.getLegalActions(self.index):
            if action == Directions.STOP:
                continue
            successor = gameState.generateSuccessor(self.index, action)
            value = self._minValue(successor, depth - 1, alpha, beta, opp_index)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        self.last_action = best_action
        return (
            best_action
            if best_action != Directions.STOP
            else self._safestLegal(gameState)
        )

    def _maxValue(self, state, depth, alpha, beta, opp_index):
        if depth == 0 or state.isOver():
            return self._evaluate(state)
        value = float("-inf")
        actions = state.getLegalActions(self.index)
        if not actions:
            return self._evaluate(state)
        for action in actions:
            succ = state.generateSuccessor(self.index, action)
            value = max(value, self._minValue(succ, depth - 1, alpha, beta, opp_index))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def _minValue(self, state, depth, alpha, beta, opp_index):
        if depth == 0 or state.isOver():
            return self._evaluate(state)
        # If we can't see the opponent, treat them as stationary
        if state.getAgentPosition(opp_index) is None:
            return self._evaluate(state)
        value = float("inf")
        actions = state.getLegalActions(opp_index)
        if not actions:
            return self._evaluate(state)
        for action in actions:
            succ = state.generateSuccessor(opp_index, action)
            value = min(value, self._maxValue(succ, depth - 1, alpha, beta, opp_index))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    # -----------------------------------------------------------------------
    # Evaluation function
    # -----------------------------------------------------------------------

    def _evaluate(self, gameState):
        """Weighted feature sum. Weights depend on role."""
        feats = self._features(gameState)
        return sum(feats.get(k, 0) * self.weights.get(k, 0) for k in feats)

    def _features(self, gameState):
        feats = util.Counter()
        my_pos = gameState.getAgentPosition(self.index)
        my_state = gameState.getAgentState(self.index)

        feats["score"] = self.getScore(gameState)

        if self.role == "offense":
            food = self.getFood(gameState).asList()
            feats["food_left"] = len(food)
            feats["carrying"] = my_state.numCarrying**0.5
            if food:
                feats["dist_to_food"] = min(
                    self.getMazeDistance(my_pos, f) for f in food
                )
            if my_state.numCarrying > 0:
                feats["dist_to_home"] = self._homeDist(my_pos) * my_state.numCarrying

            # Ghost threat
            for opp in self.getOpponents(gameState):
                opp_state = gameState.getAgentState(opp)
                opp_pos = opp_state.getPosition()
                if opp_pos is None or opp_state.isPacman:
                    continue
                d = self.getMazeDistance(my_pos, opp_pos)
                if opp_state.scaredTimer > 1:
                    feats["scared_bonus"] += max(0, 5 - d)
                else:
                    if my_state.isPacman:
                        if d <= 1:
                            feats["ghost_penalty"] = 1.0
                        elif d <= 4:
                            feats["ghost_proximity"] += 5 - d
                        # Dead-end penalty if ghost is near and we're in a tunnel
                        depth = self.map.tunnel_depth.get(my_pos, 0)
                        if depth > 0 and d < depth + 2:
                            feats["in_dead_end"] = depth

            # Capsule bonus
            capsules = self.getCapsules(gameState)
            if capsules:
                feats["capsule_bonus"] = 1.0 / (
                    1 + min(self.getMazeDistance(my_pos, c) for c in capsules)
                )

        else:  # defense
            invaders = [
                gameState.getAgentState(o)
                for o in self.getOpponents(gameState)
                if gameState.getAgentState(o).isPacman
                and gameState.getAgentPosition(o) is not None
            ]
            feats["num_invaders"] = len(invaders)
            if invaders:
                dists = [
                    self.getMazeDistance(my_pos, inv.getPosition()) for inv in invaders
                ]
                feats["dist_to_invader"] = min(dists)
            feats["on_defense"] = 0 if my_state.isPacman else 1

            # Distance to nearest choke
            if self.map.choke_points:
                feats["dist_to_choke"] = min(
                    self.getMazeDistance(my_pos, c) for c in self.map.choke_points
                )

            # Distance to recently-eaten cell
            if self.beliefs.recently_eaten:
                feats["dist_to_eaten"] = min(
                    self.getMazeDistance(my_pos, p) for p in self.beliefs.recently_eaten
                )

            defending = self.getFoodYouAreDefending(gameState).asList()
            feats["defended_food"] = len(defending)

            last_dir = (
                my_state.configuration.direction
                if my_state.configuration
                else Directions.STOP
            )
            if self.last_action == Directions.STOP:
                feats["stop_penalty"] = 1

        return feats

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _planAction(self, gameState, goal_fn, danger=False):
        my_pos = gameState.getAgentPosition(self.index)
        walls = self.map.walls
        cost_fn = self._dangerCost(gameState) if danger else None
        path = astar(walls, my_pos, goal_fn, cost_fn)
        if not path:
            return self._safestLegal(gameState)
        return path[0]

    def _dangerCost(self, gameState, weight=10.0):
        """Build a position -> cost function from current ghost beliefs."""
        threats = []
        for opp in self.getOpponents(gameState):
            opp_state = gameState.getAgentState(opp)
            if opp_state.isPacman or opp_state.scaredTimer > 1:
                continue
            pos = opp_state.getPosition() or self.beliefs.mostLikely(opp)
            if pos is not None and not self.map._isHome(pos):
                threats.append(pos)

        def cost(pos):
            c = 1.0
            for t in threats:
                d = util.manhattanDistance(pos, t)
                if d <= 2:
                    c += weight * (3 - d)
            return c

        return cost

    def _goalHome(self, pos):
        return pos in self.map.home_cells

    def _homeDist(self, pos):
        if pos in self.map.home_cells:
            return 0
        return min(self.getMazeDistance(pos, b) for b in self.map.boundary)

    def _realThreat(self, gameState, my_pos, my_state):
        if not my_state.isPacman:
            return False
        home_dist = self._homeDist(my_pos)
        for opp in self.getOpponents(gameState):
            opp_state = gameState.getAgentState(opp)
            if opp_state.isPacman or opp_state.scaredTimer > 1:
                continue
            pos = opp_state.getPosition()
            if pos is None:
                pos = self.beliefs.mostLikely(opp)
            if pos is None or self.map._isHome(pos):
                continue
            d = self.getMazeDistance(my_pos, pos)
            if d <= home_dist + 1 and d <= 4:
                return True
        return False

    def _ghostNear(self, gameState, pos, radius):
        for opp in self.getOpponents(gameState):
            opp_state = gameState.getAgentState(opp)
            if opp_state.isPacman or opp_state.scaredTimer > 1:
                continue
            opp_pos = opp_state.getPosition() or self.beliefs.mostLikely(opp)
            if opp_pos and self.getMazeDistance(pos, opp_pos) <= radius:
                return True
        return False

    def _visibleInvaders(self, gameState):
        invaders = []
        for opp in self.getOpponents(gameState):
            opp_state = gameState.getAgentState(opp)
            if opp_state.isPacman:
                pos = opp_state.getPosition()
                if pos is None:
                    pos = self.beliefs.mostLikely(opp)
                if pos is not None and self.map._isHome(pos):
                    invaders.append((opp, pos, opp_state))
        return invaders

    def _relevantOpponent(self, gameState):
        """Pick the opponent most relevant to model in minimax (closest visible, or closest believed)."""
        my_pos = gameState.getAgentPosition(self.index)
        best_opp = self.getOpponents(gameState)[0]
        best_d = float("inf")
        for opp in self.getOpponents(gameState):
            pos = gameState.getAgentPosition(opp)
            if pos is None:
                pos = self.beliefs.mostLikely(opp)
            if pos is None:
                continue
            d = self.getMazeDistance(my_pos, pos)
            if d < best_d:
                best_d = d
                best_opp = opp
        return best_opp

    def _distToOpp(self, gameState, opp):
        my_pos = gameState.getAgentPosition(self.index)
        pos = gameState.getAgentPosition(opp) or self.beliefs.mostLikely(opp)
        if pos is None:
            return float("inf")
        return self.getMazeDistance(my_pos, pos)

    def _keepDistance(self, gameState, my_pos, target_pos, desired=2):
        best = Directions.STOP
        best_score = float("inf")
        for action in gameState.getLegalActions(self.index):
            dx, dy = Actions.directionToVector(action)
            nx, ny = int(my_pos[0] + dx), int(my_pos[1] + dy)
            if not self.map._isHome((nx, ny)):
                continue
            d = self.getMazeDistance((nx, ny), target_pos)
            score = abs(d - desired)
            if score < best_score:
                best_score = score
                best = action
        return best

    def _safestLegal(self, gameState):
        my_pos = gameState.getAgentPosition(self.index)
        actions = gameState.getLegalActions(self.index)
        non_stop = [a for a in actions if a != Directions.STOP]
        if non_stop:
            actions = non_stop
        if not actions:
            return Directions.STOP
        last_dir = gameState.getAgentState(self.index).configuration.direction
        reverse = Directions.REVERSE.get(last_dir, Directions.STOP)
        cost_fn = self._dangerCost(gameState)

        def score(action):
            dx, dy = Actions.directionToVector(action)
            nx, ny = int(my_pos[0] + dx), int(my_pos[1] + dy)
            s = cost_fn((nx, ny))
            if action == reverse:
                s += 2.0
            return s

        return min(actions, key=score)
