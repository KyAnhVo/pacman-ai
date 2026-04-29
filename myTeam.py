from __future__ import annotations

from collections import deque

from capture import SIGHT_RANGE, GameState
from captureAgents import CaptureAgent
from game import Actions, Directions
from layout import Layout
from util import PriorityQueue, manhattanDistance


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="GreedyPointAgent",
    second="InterceptDefenseAgent",
):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


########################
# Enemy Belief Tracker #
########################


class EnemyBeliefTracker:
    """
    Tracks the belief distribution of enemy positions based on the modified
    capture's noise system (state.noise[enemy] gives a scrambled Configuration
    with pos in [true ± 2] per axis when out of sight).

    Note: this belief tracker works the best when allies are further away from each other, thus the
    intersected area is small due to intersection.
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
            2. Define the square ring defining the possible position of the cell given
                the by the enemyDistance (with noise), then intersect the square ring with
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
        noise_list = getattr(gameState, "noise", None)
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

    Adds a target tracker so we don't flip-flop between equidistant food in
    open spaces. Commits to a food target until we either reach it, the target
    gets eaten, or we go STALE_THRESHOLD turns without making progress toward it.
    """

    # ------ Tunables ------
    CARRY_THRESHOLD = 5  # carry this many before considering home
    DANGER_RADIUS = 4  # within this maze distance an enemy ghost is "near"
    MINIMAX_DEPTH = 4  # plies (each ply = one of our + one ghost move)
    SAFE_FOOD_RADIUS = 6  # food considered "close" for safety check
    SAFE_FOOD_MARGIN = 1  # ghost_d - my_d must be >= this (was 2, now looser)
    CAPSULE_RUSH_RADIUS = 7  # capsule considered "close" enough to rush

    # Belief-as-threat handling
    BELIEF_DIFFUSE_LIMIT = (
        50  # if belief set bigger than this, ghost is effectively unknown
    )

    # Target tracker
    STALE_THRESHOLD = 5  # turns without progress before giving up on a target
    BLACKLIST_TTL = 30  # turns to avoid a previously-stuck target
    DEBUG = False  # set True to print branch decisions

    def registerInitialState(self, gameState: GameState):
        CaptureAgent.registerInitialState(self, gameState)

        # Walls / dimensions
        self.walls = gameState.getWalls()
        self.width = self.walls.width
        self.height = self.walls.height

        # Capsules (initial — for reference; we always read fresh in chooseAction)
        self.capsules = gameState.getCapsules()

        # Midline
        self.x_mid = self.walls.width // 2

        # Opponent indices
        self.enemy_indices = self.getOpponents(gameState)

        # Belief trackers — one per enemy
        self.belief_trackers = {
            enemy: EnemyBeliefTracker(self.index, enemy, gameState, naive=True)
            for enemy in self.enemy_indices
        }

        # Home boundary cells
        self.start = gameState.getAgentPosition(self.index)
        self.home_x = (self.width // 2) - 1 if self.red else (self.width // 2)
        self.home_boundary = [
            (self.home_x, y)
            for y in range(self.height)
            if not self.walls[self.home_x][y]
        ]

        # Precompute distance-to-home grid (BFS from boundary, multi-source)
        self._home_dist_grid = self._computeHomeDistGrid()

        # Target tracker state
        self.current_target = None
        self.target_progress_dist = None
        self.target_age = 0
        self.target_blacklist = {}  # {food_pos: turns remaining}

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

        # Decay the blacklist each turn
        self.target_blacklist = {
            f: t - 1 for f, t in self.target_blacklist.items() if t > 1
        }

        # Threat assessment
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
            or (carrying > 0 and time_left < home_dist + 20)
        )

        # Branch 1: enemy nearby
        if in_danger:
            self._dbg("in_danger=True, carry=%d" % carrying)
            self._clearTarget()  # don't commit while fleeing

            # Capsule close? rush it via minimax (BEFORE going home — capsule is better)
            if capsules:
                nearest_cap = min(
                    capsules, key=lambda c: self.getMazeDistance(my_pos, c)
                )
                cap_dist = self.getMazeDistance(my_pos, nearest_cap)
                if cap_dist <= self.CAPSULE_RUSH_RADIUS and cap_dist <= home_dist:
                    self._dbg("rush capsule @ %s" % str(nearest_cap))
                    action = self._minimaxEscape(
                        gameState,
                        prefer="capsule",
                        threat_idx=threat_idx,
                        capsule=nearest_cap,
                    )
                    if action is not None:
                        return action

            # Carrying enough -> minimax home
            if carrying >= self.CARRY_THRESHOLD or must_return:
                self._dbg("carry enough, head home")
                action = self._minimaxEscape(
                    gameState, prefer="home", threat_idx=threat_idx
                )
                if action is not None:
                    return action

            # Try a close, safe food
            if food_list:
                close_food = [
                    f
                    for f in food_list
                    if self.getMazeDistance(my_pos, f) <= self.SAFE_FOOD_RADIUS
                ]
                if close_food:
                    safe = self._safestFood(gameState, my_pos, close_food, threat_pos)
                    if safe is not None:
                        self._dbg("eat safe food @ %s" % str(safe))
                        action = self._aStarFirstAction(
                            gameState,
                            my_pos,
                            lambda s: s == safe,
                            danger_fn=self._buildDangerFn(gameState),
                        )
                        if action is not None:
                            return action

            # Default: rush home with minimax
            self._dbg("default: rush home")
            action = self._minimaxEscape(
                gameState, prefer="home", threat_idx=threat_idx
            )
            if action is not None:
                return action

        # Branch 2: no close enemy
        if must_return and carrying > 0:
            self._dbg("must_return, carry=%d" % carrying)
            self._clearTarget()
            action = self._aStarFirstAction(
                gameState,
                my_pos,
                self._isHome,
                danger_fn=self._buildDangerFn(gameState),
            )
            if action is not None:
                return action

        # Greedy food with target commitment
        if food_list:
            target = self._selectFoodTarget(my_pos, food_list)
            if target is not None:
                self._dbg("target=%s age=%d" % (str(target), self.target_age))
                action = self._aStarFirstAction(
                    gameState,
                    my_pos,
                    lambda s: s == target,
                    danger_fn=self._buildDangerFn(gameState, weight=4.0),
                )
                if action is not None:
                    return action

        # Fallback: any safe legal move
        return self._safestLegal(gameState, my_pos)

    # -------------------- target tracker --------------------

    def _selectFoodTarget(self, my_pos, food_list):
        """
        Commit to a food target across turns. Drop the target if eaten or stale.
        Stale = STALE_THRESHOLD turns with no improvement on best-distance-so-far.
        """
        food_set = set(food_list)

        # 1) Validate current target
        if self.current_target is not None:
            if self.current_target not in food_set:
                # eaten by us or someone else
                self._clearTarget()
            else:
                dist_now = self.getMazeDistance(my_pos, self.current_target)
                if (
                    self.target_progress_dist is None
                    or dist_now < self.target_progress_dist
                ):
                    self.target_progress_dist = dist_now
                    self.target_age = 0
                else:
                    self.target_age += 1

                if self.target_age >= self.STALE_THRESHOLD:
                    # blacklist and pick fresh
                    self.target_blacklist[self.current_target] = self.BLACKLIST_TTL
                    self._clearTarget()

        # 2) Pick a new target if needed
        if self.current_target is None:
            candidates = [f for f in food_list if f not in self.target_blacklist]
            if not candidates:
                # Everything blacklisted — clear and use everything
                self.target_blacklist.clear()
                candidates = food_list
            if candidates:
                # Score: prefer close food, but penalize ambiguous foods (lots of
                # other foods nearby — those tend to cause flip-flop)
                def score(f):
                    d = self.getMazeDistance(my_pos, f)
                    ambiguity = sum(
                        1
                        for other in food_list
                        if other != f and manhattanDistance(other, f) <= 2
                    )
                    return d + 0.5 * ambiguity

                self.current_target = min(candidates, key=score)
                self.target_progress_dist = self.getMazeDistance(
                    my_pos, self.current_target
                )
                self.target_age = 0

        return self.current_target

    def _clearTarget(self):
        self.current_target = None
        self.target_progress_dist = None
        self.target_age = 0

    # -------------------- threat helpers --------------------

    def _nearestThreat(self, gameState, my_pos):
        """Return (maze_dist, pos, enemy_idx) of nearest dangerous ghost, or (None, None, None)."""
        best = (None, None, None)
        for enemy in self.enemy_indices:
            es = gameState.getAgentState(enemy)
            if es.isPacman:
                continue
            if es.scaredTimer > 1:
                continue
            exact = gameState.getAgentPosition(enemy)
            if exact is None:
                # Belief fallback only if belief is tight enough to be useful
                belief = self.belief_trackers[enemy].enemy_position_distribution
                if not belief or len(belief) > self.BELIEF_DIFFUSE_LIMIT:
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
        """
        Cost overlay: positions adjacent to threatening ghosts cost more.
        Only uses confirmed positions and tight beliefs — diffuse beliefs
        contaminate the cost field uniformly and cause flip-flop.
        """
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
                # Only contribute if belief is tight enough to be informative
                if 0 < len(belief) <= self.BELIEF_DIFFUSE_LIMIT:
                    threats.extend(belief)
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

    ##########
    # A* #####
    ##########

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
                    # h=0 (Dijkstra) — admissible & cheap. Could add Manhattan
                    # if pathing becomes a bottleneck.
                    frontier.push((succ, path + [action]), new_g)
        return None

    ###################
    # Minimax to run ##
    ###################

    def _minimaxEscape(self, gameState, prefer, threat_idx, capsule=None):
        """
        Lightweight minimax: we maximize, threat ghost minimizes.
        Search MINIMAX_DEPTH plies (one ply = one of our move + one ghost move).
        """
        my_pos = gameState.getAgentPosition(self.index)
        if threat_idx is None:
            return None

        ghost_pos = gameState.getAgentPosition(threat_idx)
        if ghost_pos is None:
            belief = self.belief_trackers[threat_idx].enemy_position_distribution
            if not belief or len(belief) > self.BELIEF_DIFFUSE_LIMIT:
                return None
            ghost_pos = min(belief, key=lambda p: self.getMazeDistance(my_pos, p))
            ghost_pos = (int(ghost_pos[0]), int(ghost_pos[1]))

        carrying = gameState.getAgentState(self.index).numCarrying

        def eval_state(me, ghost):
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
            # Caught: very bad — scaled by what we'd lose
            if ghost_d == 0:
                return -1e6 - 1000 * carrying
            score = -goal_d * 10 + min(ghost_d, 6) * 4
            if self._isHome(me):
                score += 50
            if prefer == "capsule" and capsule is not None and me == capsule:
                score += 200
            return score

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
            # Capture ghosts can stop and reverse — same action set as us minus STOP
            # (a stopping ghost is rarely optimal vs a fleeing pacman)
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
                return eval_state(me, ghost), None
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
                return eval_state(me, ghost), None
            best_v = float("inf")
            for new_ghost in ghost_moves(ghost):
                v, _ = maxnode(me, new_ghost, depth - 1, alpha, beta)
                if v < best_v:
                    best_v = v
                beta = min(beta, best_v)
                if beta <= alpha:
                    break
            return best_v, None

        # MINIMAX_DEPTH plies = MINIMAX_DEPTH * 2 half-plies in the search
        _, action = maxnode(
            my_pos, ghost_pos, self.MINIMAX_DEPTH * 2, float("-inf"), float("inf")
        )
        if action is None:
            return None
        legal = gameState.getLegalActions(self.index)
        if action not in legal:
            return None
        return action

    # -------------------- safe food check --------------------

    def _safestFood(self, gameState, my_pos, foods, threat_pos):
        """
        Pick a food we can reach with a real cushion vs the ghost.
        Margin loosened from 2 to 1 so we don't over-reject in open spaces.
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
            margin = ghost_d - my_d
            if margin >= self.SAFE_FOOD_MARGIN:
                # Bigger weight on distance — closer food is much better
                score = margin - my_d * 0.5
                if score > best_score:
                    best_score = score
                    best = f
        return best

    ################
    # Helpers ######
    ################

    def _isHome(self, pos):
        x = int(pos[0])
        if self.red:
            return x <= self.home_x
        return x >= self.home_x

    def _distToHome(self, pos):
        ix, iy = int(pos[0]), int(pos[1])
        if self._isHome(pos):
            return 0
        d = self._home_dist_grid[ix][iy]
        return d if d is not None else 0

    def _computeHomeDistGrid(self):
        """Multi-source BFS from all home-boundary cells. Returns 2D list of dists (or None)."""
        grid = [[None] * self.height for _ in range(self.width)]
        q = deque()
        for bx, by in self.home_boundary:
            grid[bx][by] = 0
            q.append((bx, by))
        while q:
            x, y = q.popleft()
            d = grid[x][y]
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self.walls[nx][ny]:
                    continue
                if grid[nx][ny] is None:
                    grid[nx][ny] = d + 1
                    q.append((nx, ny))
        return grid

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

    def _dbg(self, msg):
        if self.DEBUG:
            print("[%d] %s" % (self.index, msg))


#######################################################################
################# FROM HERE IT'S THE DEFENSE AGENT ####################
#######################################################################


###########################
# Entry point calculation #
###########################


class EntryPoints:
    """
    Computes natural intercept points for defense.

    For each food cluster, we walk the invader's shortest path from the
    boundary to that cluster's medoid, and find the first cell along the
    path that's a real choke -- i.e., removing it forces a
    detour. That cell is the intercept point: the closest place to the
    boundary where blocking the invader matters.
    """

    FOOD_CLUSTER_WEIGHT_DIST = 2  # foods within this maze-dist cluster together
    MIN_CHOKE_GAIN = 3  # detour increase that qualifies as a real choke
    MAX_PATH_DEPTH = 8  # don't walk further than this from boundary

    def __init__(self, layout: Layout, game_state: GameState, is_red: bool):
        self.layout = layout
        self.is_red = is_red
        self.walls = layout.walls
        self.width = layout.width
        self.height = layout.height

        # All protected food on our side.
        self.foods = set(
            (game_state.getRedFood() if is_red else game_state.getBlueFood()).asList()
        )

        # Home-side cells: non-wall cells on our half of the board.
        mid_x = self.width // 2
        if is_red:
            self.home_x_range = range(0, mid_x)
            self.boundary_x = mid_x - 1
        else:
            self.home_x_range = range(mid_x, self.width)
            self.boundary_x = mid_x

        self.home_cells: set = {
            (x, y)
            for x in self.home_x_range
            for y in range(self.height)
            if not self.walls[x][y]
        }

        self.boundary_cells: list = [
            (self.boundary_x, y)
            for y in range(self.height)
            if not self.walls[self.boundary_x][y]
        ]

        # Group foods, then compute entries.
        self.food_clusters: list = self.group_foods()
        self.cluster_reps: list = [self._medoid(c) for c in self.food_clusters]

        # entry -> (protected_rep, detour_gain)
        self.entry_to_protected_food: dict = {}
        self.entryPoints: set = self.calculate_entries()

    # -------------------- food grouping --------------------

    def group_foods(self) -> list:
        """Single-link cluster by maze distance."""
        if not self.foods:
            return []
        food_list = list(self.foods)
        n = len(food_list)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                if (
                    manhattanDistance(food_list[i], food_list[j])
                    > self.FOOD_CLUSTER_WEIGHT_DIST
                ):
                    continue
                d = self._maze_dist(food_list[i], food_list[j], blocked=set())
                if d is not None and d <= self.FOOD_CLUSTER_WEIGHT_DIST:
                    union(i, j)

        groups: dict = {}
        for i, f in enumerate(food_list):
            groups.setdefault(find(i), []).append(f)
        return list(groups.values())

    def _medoid(self, cluster: list):
        """Cluster member minimizing total distance to others."""
        if len(cluster) == 1:
            return cluster[0]
        best, best_total = None, float("inf")
        for candidate in cluster:
            total = 0
            for other in cluster:
                if other == candidate:
                    continue
                d = self._maze_dist(candidate, other, blocked=set())
                if d is None:
                    total = float("inf")
                    break
                total += d
            if total < best_total:
                best_total, best = total, candidate
        return best if best is not None else cluster[0]

    # -------------------- entry-point calculation --------------------

    def calculate_entries(self) -> set:
        """
        For each cluster rep, find the natural intercept point on the
        invader's shortest path from boundary to that rep.
        """
        entries = set()
        if not self.cluster_reps or not self.boundary_cells:
            return entries

        for rep in self.cluster_reps:
            choke = self._findChokeOnPath(rep)
            if choke is not None:
                entries.add(choke)
                # Also record what this entry protects and by how much
                base_d = self._multi_source_dist(
                    self.boundary_cells, rep, blocked=set()
                )
                blocked_d = self._multi_source_dist(
                    self.boundary_cells, rep, blocked={choke}
                )
                if base_d is not None and blocked_d is not None:
                    self.entry_to_protected_food[choke] = (rep, blocked_d - base_d)

        return entries

    def _findChokeOnPath(self, rep):
        """
        Walk the boundary->rep shortest path. Return the first cell whose
        removal increases the detour by >= MIN_CHOKE_GAIN, or the path
        midpoint as fallback if no real choke exists within MAX_PATH_DEPTH.
        """
        path = self._shortestPath(self.boundary_cells, rep)
        if not path:
            return None

        base_d = self._multi_source_dist(self.boundary_cells, rep, blocked=set())
        if base_d is None:
            return None

        # Walk the path from boundary toward the rep. Skip the boundary
        # itself and the rep itself.
        for i, cell in enumerate(path):
            if cell in self.boundary_cells or cell == rep:
                continue
            if i > self.MAX_PATH_DEPTH:
                break
            new_d = self._multi_source_dist(self.boundary_cells, rep, blocked={cell})
            # Either fully disconnected (perfect choke) or large detour
            if new_d is None or new_d - base_d >= self.MIN_CHOKE_GAIN:
                return cell

        # Fallback: no clear choke found on the path. Use a cell ~2 steps
        # from boundary as a default patrol point.
        for i, cell in enumerate(path):
            if cell in self.boundary_cells or cell == rep:
                continue
            if i >= 2:
                return cell

        return None

    def _shortestPath(self, sources: list, dst):
        """BFS from any source to dst, returning the path as a list of cells."""
        if dst in sources:
            return [dst]
        prev = {}
        seen = set()
        q = deque()
        for s in sources:
            q.append(s)
            seen.add(s)
            prev[s] = None
        while q:
            cur = q.popleft()
            if cur == dst:
                # Reconstruct path
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path
            x, y = cur
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self.walls[nx][ny] or (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                prev[(nx, ny)] = cur
                q.append((nx, ny))
        return []

    # -------------------- BFS helpers --------------------

    def _maze_dist(self, src, dst, blocked: set):
        """BFS on home-side cells, treating blocked cells as walls."""
        if src == dst:
            return 0
        if src in blocked or dst in blocked:
            return None
        if src not in self.home_cells or dst not in self.home_cells:
            return self._bfs_full_board(src, dst, blocked)

        q = deque([(src, 0)])
        seen = {src}
        while q:
            (x, y), d = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in seen or (nx, ny) in blocked:
                    continue
                if (nx, ny) not in self.home_cells:
                    continue
                if (nx, ny) == dst:
                    return d + 1
                seen.add((nx, ny))
                q.append(((nx, ny), d + 1))
        return None

    def _bfs_full_board(self, src, dst, blocked: set):
        if src == dst:
            return 0
        q = deque([(src, 0)])
        seen = {src}
        while q:
            (x, y), d = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self.walls[nx][ny] or (nx, ny) in blocked or (nx, ny) in seen:
                    continue
                if (nx, ny) == dst:
                    return d + 1
                seen.add((nx, ny))
                q.append(((nx, ny), d + 1))
        return None

    def _multi_source_dist(self, sources: list, dst, blocked: set):
        """BFS from all sources simultaneously to dst, with blocked cells removed."""
        if dst in blocked:
            return None
        q = deque()
        seen = set()
        for s in sources:
            if s in blocked:
                continue
            if s == dst:
                return 0
            q.append((s, 0))
            seen.add(s)
        while q:
            (x, y), d = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self.walls[nx][ny] or (nx, ny) in blocked or (nx, ny) in seen:
                    continue
                if (nx, ny) == dst:
                    return d + 1
                seen.add((nx, ny))
                q.append(((nx, ny), d + 1))
        return None


##################
# Defense Agent  #
##################


class InterceptDefenseAgent(CaptureAgent):
    """
    State machine defender:

      PATROL    -- no invader signal. Cycle through entry points (top-to-
                   bottom by y-coord), dwelling DWELL_TURNS at each. Recent
                   eaten food overrides the cycle and routes us to the
                   entry closest to the eaten cell.
      INTERCEPT -- invader detected, race to the entry that cuts their
                   shortest path home, then hold it.
      RETREAT   -- we're scared (capsule eaten), keep distance from invader
                   while staying near entry points so we can re-engage when
                   the timer runs down.

    Invader localization (in priority order):
      1. Exact position if visible.
      2. Recently-eaten food cell (within EATEN_FRESH_TURNS).
      3. Belief tracker if tight enough.
    """

    # ------ Tunables ------
    DANGER_RADIUS = 3  # invader within this maze-dist = imminent threat
    EATEN_FRESH_TURNS = 6  # remember an eaten cell for this many turns
    SCARED_BUFFER = 2  # keep this much distance when scared
    BELIEF_DIFFUSE_LIMIT = 50  # ignore belief if larger than this
    DWELL_TURNS = 4  # turns to spend at each patrol entry

    DEBUG = False

    def registerInitialState(self, gameState: GameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.walls = gameState.getWalls()
        self.width = self.walls.width
        self.height = self.walls.height

        self.start = gameState.getAgentPosition(self.index)
        self.enemy_indices = self.getOpponents(gameState)

        # Belief trackers, one per enemy.
        self.belief_trackers = {
            enemy: EnemyBeliefTracker(self.index, enemy, gameState, naive=True)
            for enemy in self.enemy_indices
        }

        # Entry-point analysis.
        self.entry_analysis = EntryPoints(gameState.data.layout, gameState, self.red)

        # Home boundary cells
        self.home_x = (self.width // 2) - 1 if self.red else (self.width // 2)
        self.home_boundary = [
            (self.home_x, y)
            for y in range(self.height)
            if not self.walls[self.home_x][y]
        ]

        # Patrol route: entries sorted top-to-bottom by y. If no entries
        # found, fall back to a thinned boundary patrol (every 3rd cell).
        if self.entry_analysis.entryPoints:
            self.patrol_route = sorted(
                self.entry_analysis.entryPoints, key=lambda c: c[1]
            )
        else:
            self.patrol_route = self.home_boundary[::3] or self.home_boundary
        self.patrol_idx = 0
        self.dwell_counter = 0  # turns spent at current patrol target

        # Track food state across turns.
        defending = self.getFoodYouAreDefending(gameState).asList()
        self.prev_defending_food = set(defending)

        # Recent eaten cells with turn-decay timestamps.
        self.recent_eaten: dict = {}  # {pos: turns_remaining}
        self.turn = 0

    # -------------------- main entry --------------------

    def chooseAction(self, gameState: GameState):
        self.turn += 1

        # Update beliefs.
        for tracker in self.belief_trackers.values():
            tracker.update(gameState)

        # Detect newly eaten food.
        defending_now = set(self.getFoodYouAreDefending(gameState).asList())
        newly_eaten = self.prev_defending_food - defending_now
        for cell in newly_eaten:
            self.recent_eaten[cell] = self.EATEN_FRESH_TURNS
        self.prev_defending_food = defending_now

        # Decay recent_eaten.
        self.recent_eaten = {c: t - 1 for c, t in self.recent_eaten.items() if t > 1}

        my_pos = gameState.getAgentPosition(self.index)
        my_state = gameState.getAgentState(self.index)
        is_scared = my_state.scaredTimer > 1

        # Localize invaders.
        invaders = self._localizeInvaders(gameState, my_pos)
        real_invaders = [
            inv for inv in invaders if self._shouldChase(my_pos, inv[1], inv[2])
        ]

        # Pick state.
        if is_scared and real_invaders:
            return self._retreat(gameState, my_pos, invaders)
        elif real_invaders:
            return self._intercept(gameState, my_pos, invaders)
        else:
            return self._patrol(gameState, my_pos)

    # -------------------- invader localization --------------------

    def _localizeInvaders(self, gameState, my_pos):
        """Return list of (enemy_idx, position, confidence) for each Pacman invader."""
        results = []
        for enemy in self.enemy_indices:
            es = gameState.getAgentState(enemy)
            if not es.isPacman:
                continue

            exact = gameState.getAgentPosition(enemy)
            if exact is not None:
                results.append((enemy, exact, "exact"))
                continue

            if self.recent_eaten:
                pos = max(self.recent_eaten.items(), key=lambda kv: kv[1])[0]
                results.append((enemy, pos, "eaten"))
                continue

            belief = self.belief_trackers[enemy].enemy_position_distribution
            if belief and len(belief) <= self.BELIEF_DIFFUSE_LIMIT:
                home_side_belief = [p for p in belief if self._isOurSide(p)]
                if home_side_belief:
                    pos = min(
                        home_side_belief, key=lambda p: self.getMazeDistance(my_pos, p)
                    )
                    results.append((enemy, pos, "belief"))

        return results

    # -------------------- INTERCEPT state --------------------

    def _intercept(self, gameState, my_pos, invaders):
        """Race to the entry point on invader's escape path, then hold."""
        # Reset dwell counter so we don't immediately dwell on returning to PATROL.
        self.dwell_counter = 0

        primary = min(invaders, key=lambda inv: self._distToOurBoundary(inv[1]))
        inv_idx, inv_pos, conf = primary
        self._dbg("INTERCEPT inv=%s at %s (%s)" % (inv_idx, str(inv_pos), conf))

        # Point-blank visible: just chase.
        if conf == "exact":
            inv_dist = self.getMazeDistance(my_pos, inv_pos)
            if inv_dist <= self.DANGER_RADIUS:
                return self._stepToward(gameState, my_pos, inv_pos)

        target = self._bestInterceptCell(my_pos, inv_pos)
        if target is None:
            return self._stepToward(gameState, my_pos, inv_pos)

        # Already on intercept cell -- pressure the invader by stepping toward them.
        if my_pos == target:
            return self._stepToward(gameState, my_pos, inv_pos)

        return self._stepToward(gameState, my_pos, target)

    def _bestInterceptCell(self, my_pos, inv_pos):
        """Among entries, pick the one we can reach before invader and that's on their path."""
        candidates = list(self.entry_analysis.entryPoints) or self.home_boundary

        best = None
        best_score = float("-inf")
        for cell in candidates:
            try:
                inv_to_cell = self.getMazeDistance(inv_pos, cell)
                me_to_cell = self.getMazeDistance(my_pos, cell)
            except Exception:
                continue
            entry_value = self.entry_analysis.entry_to_protected_food.get(
                cell, (None, 0)
            )[1]
            race_margin = inv_to_cell - me_to_cell
            on_path_bonus = -inv_to_cell * 0.5

            if race_margin < -2:
                continue

            score = race_margin * 2 + on_path_bonus + entry_value * 0.5
            if score > best_score:
                best_score = score
                best = cell
        return best

    # -------------------- RETREAT state --------------------

    def _retreat(self, gameState, my_pos, invaders):
        """Maintain distance from invader while staying near a patrol target."""
        primary = min(invaders, key=lambda inv: self.getMazeDistance(my_pos, inv[1]))
        inv_pos = primary[1]
        inv_dist = self.getMazeDistance(my_pos, inv_pos)
        self._dbg("RETREAT inv at %s d=%d" % (str(inv_pos), inv_dist))

        actions = gameState.getLegalActions(self.index)
        target_entry = self._currentPatrolTarget(my_pos)

        def score(action):
            dx, dy = Actions.directionToVector(action)
            nx, ny = int(my_pos[0] + dx), int(my_pos[1] + dy)
            if not self._isOurSide((nx, ny)):
                return float("-inf")
            try:
                d_inv = self.getMazeDistance((nx, ny), inv_pos)
                d_entry = (
                    self.getMazeDistance((nx, ny), target_entry) if target_entry else 0
                )
            except Exception:
                return float("-inf")
            buffer_term = min(d_inv, self.SCARED_BUFFER + 2) * 5
            entry_term = -d_entry * 1
            stop_pen = -3 if action == Directions.STOP else 0
            return buffer_term + entry_term + stop_pen

        return max(actions, key=score)

    # -------------------- PATROL state --------------------

    def _patrol(self, gameState, my_pos):
        """Cycle through entries with dwell. Recent eaten food overrides cycle."""
        target = self._currentPatrolTarget(my_pos)
        if target is None:
            target = min(
                self.home_boundary, key=lambda b: self.getMazeDistance(my_pos, b)
            )
        self._dbg(
            "PATROL idx=%d dwell=%d target=%s"
            % (self.patrol_idx, self.dwell_counter, str(target))
        )

        # Reached target?
        if my_pos == target:
            self.dwell_counter += 1
            if self.dwell_counter >= self.DWELL_TURNS:
                # Advance to next entry on the route.
                self.dwell_counter = 0
                self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_route)
                target = self.patrol_route[self.patrol_idx]
                return self._stepToward(gameState, my_pos, target)
            # Still dwelling: hold position.
            return Directions.STOP

        # Not at target yet: walk toward it. Reset dwell counter (we're traveling).
        self.dwell_counter = 0
        return self._stepToward(gameState, my_pos, target)

    def _currentPatrolTarget(self, my_pos):
        """
        Return the cell we're currently patrolling toward.

        Priority:
          1. If food was recently eaten, divert to the entry closest to the
             eaten cell (and resync patrol_idx so the cycle resumes from there).
          2. Otherwise, the route cell at patrol_idx.
        """
        if not self.patrol_route:
            return None

        # Eaten-food override.
        if self.recent_eaten:
            most_recent = max(self.recent_eaten.items(), key=lambda kv: kv[1])[0]
            target = min(
                self.patrol_route,
                key=lambda c: self.getMazeDistance(c, most_recent),
            )
            new_idx = self.patrol_route.index(target)
            if new_idx != self.patrol_idx:
                # Diverting to a new entry -- reset dwell.
                self.dwell_counter = 0
                self.patrol_idx = new_idx
            return target

        return self.patrol_route[self.patrol_idx]

    # -------------------- step helper --------------------

    def _stepToward(self, gameState, my_pos, target):
        actions = gameState.getLegalActions(self.index)
        if not actions:
            return Directions.STOP

        best, best_d = None, float("inf")
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            nx, ny = int(my_pos[0] + dx), int(my_pos[1] + dy)
            if not self._isOurSide((nx, ny)):
                continue
            try:
                d = self.getMazeDistance((nx, ny), target)
            except Exception:
                d = manhattanDistance((nx, ny), target)
            if action == Directions.STOP:
                d += 0.5
            if d < best_d:
                best_d = d
                best = action

        if best is None:
            return actions[0]
        return best

    def _shouldChase(self, my_pos, inv_pos, conf):
        # Always chase if we're definitely going to win the race
        me_to_inv = self.getMazeDistance(my_pos, inv_pos)
        if me_to_inv <= 2:
            return True

        # Check if they're probably bait
        inv_depth = self._distToOurBoundary(inv_pos)  # how far from THEIR home
        if inv_depth <= 2 and not self.recent_eaten:
            return False  # shallow + hasn't eaten = bait

        # Check the race honestly
        inv_to_their_home = self._distToOurBoundary(inv_pos)
        if inv_to_their_home < me_to_inv - 1:
            return False  # they can escape before we arrive

        return True

    # -------------------- side helpers --------------------

    def _isOurSide(self, pos):
        x = int(pos[0])
        if self.red:
            return x <= self.home_x
        return x >= self.home_x

    def _distToOurBoundary(self, pos):
        if not self.home_boundary:
            return 0
        try:
            return min(self.getMazeDistance(pos, b) for b in self.home_boundary)
        except Exception:
            return min(manhattanDistance(pos, b) for b in self.home_boundary)

    def _dbg(self, msg):
        if self.DEBUG:
            print("[D%d] %s" % (self.index, msg))
