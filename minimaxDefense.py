from __future__ import annotations

import random
from collections import deque

from capture import SIGHT_RANGE, GameState
from captureAgents import CaptureAgent
from game import Actions, Directions, Grid
from greedyBastards import DefenseAgent
from layout import Layout
from util import PriorityQueue, manhattanDistance

###########################
# Entry point calculation #
###########################


class EntryPoints:
    """
    Computes structural defensive entry points for our home side.

    Pipeline:
      1. Group protected foods into clusters (foods within FOOD_CLUSTER_WEIGHT_DIST
         maze-distance of each other end up in the same cluster). Each cluster is
         represented by its medoid (the food minimizing total distance to others
         in the cluster -- always a legal cell, unlike the COM).
      2. Greedily pick cells whose removal maximizes the average maze distance
         from the boundary to those cluster medoids. Stop when removing another
         cell disconnects the graph, when we've hit MAX_ENTRIES, or when the
         marginal gain falls below MIN_GAIN.

    Treated as a one-shot heuristic computed at startup. Map topology doesn't
    change, so entries stay structurally meaningful even as food gets eaten.
    """

    FOOD_CLUSTER_WEIGHT_DIST = 2  # foods within this maze-dist cluster together
    MAX_ENTRIES = 6  # hard cap so open maps don't run forever
    MIN_GAIN = 0.5  # stop early if greedy step doesn't help much

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
        # Boundary x is the column adjacent to the midline on our side.
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

        # Boundary cells (where enemies cross over).
        self.boundary_cells: list = [
            (self.boundary_x, y)
            for y in range(self.height)
            if not self.walls[self.boundary_x][y]
        ]

        # Group foods, then compute entries.
        self.food_clusters: list = self.group_foods()
        self.cluster_reps: list = [self._medoid(c) for c in self.food_clusters]
        self.entryPoints: set = self.calculate_entries()

    # -------------------- food grouping --------------------

    def group_foods(self) -> list:
        """
        Single-link cluster foods by maze distance: two foods are in the same
        cluster if there's a chain of foods between them where consecutive
        members are within FOOD_CLUSTER_WEIGHT_DIST.

        Returns a list of clusters (each cluster is a list of food positions).
        """
        if not self.foods:
            return []

        food_list = list(self.foods)
        n = len(food_list)
        # Union-find
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

        # Pairwise: maze distance is expensive, but food count is small (~30).
        # Cheap upper bound via Manhattan first.
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
        """Return the cluster member minimizing total maze distance to others."""
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
        Greedy: each iteration, pick the home-side cell whose removal most
        increases the average maze distance from the boundary to cluster reps.
        If a candidate fully disconnects boundary from any rep, take it
        immediately (infinite gain) and stop.
        """
        if not self.cluster_reps or not self.boundary_cells:
            return set()

        entries: set = set()
        # Cells we won't consider removing: the reps themselves, the boundary.
        protected = set(self.cluster_reps) | set(self.boundary_cells)

        # Baseline cost: avg dist from boundary to each rep with nothing blocked.
        base_cost = self._avg_boundary_to_reps(blocked=set())
        if base_cost is None:
            return set()  # graph already disconnected somehow

        for _ in range(self.MAX_ENTRIES):
            best_cell = None
            best_gain = self.MIN_GAIN
            disconnect_cell = None

            candidates = self.home_cells - entries - protected

            for cell in candidates:
                blocked = entries | {cell}
                cost = self._avg_boundary_to_reps(blocked=blocked)
                if cost is None:
                    # This cell disconnects the graph -- take it immediately.
                    disconnect_cell = cell
                    break
                gain = cost - base_cost
                if gain > best_gain:
                    best_gain = gain
                    best_cell = cell

            if disconnect_cell is not None:
                entries.add(disconnect_cell)
                break

            if best_cell is None:
                break  # no candidate beats the gain threshold

            entries.add(best_cell)
            base_cost = base_cost + best_gain

        return entries

    # -------------------- BFS helpers --------------------

    def _maze_dist(self, src, dst, blocked: set):
        """BFS on home-side cells, treating `blocked` cells as walls. None if unreachable."""
        if src == dst:
            return 0
        if src in blocked or dst in blocked:
            return None
        if src not in self.home_cells or dst not in self.home_cells:
            # Fall back to full-board BFS; some reps/boundaries might be on edge.
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
        """Fallback BFS over all non-wall cells."""
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

    def _avg_boundary_to_reps(self, blocked: set):
        """
        Average over each (boundary cell, rep) pair of the min maze distance.
        Returns None if any rep is unreachable from any boundary cell -- which
        means `blocked` cuts off some food, i.e. perfect defense.
        """
        if not self.boundary_cells or not self.cluster_reps:
            return 0.0

        total = 0.0
        count = 0
        for rep in self.cluster_reps:
            # Multi-source BFS from all boundary cells, treating blocked as walls.
            d = self._multi_source_dist(self.boundary_cells, rep, blocked)
            if d is None:
                return None  # disconnected
            total += d
            count += 1
        return total / count if count else 0.0

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
