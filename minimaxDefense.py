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
    def __init__(self, layout: Layout, game_state: GameState, is_red: bool):
        self.FOOD_CLUSTER_MIN_MAX_DIST = 3
        self.layout = layout
        self.foods = set(
            (game_state.getRedFood() if is_red else game_state.getBlueFood()).asList()
        )
        self.entryPoints: set = self.calculate_entries()

    def group_foods(self):
        """
        Group foods together, close within FOOD_CLUSTER_MINI_MAX_DIST
        """
        for pos in self.foods:
            pass

    def calculate_entries(self):
        """
        Calculate the entry point set greedily.
        Greedy heuristic: choose the cell that forces the highest increase in average
        distance to the food groups
        """
        return set()
