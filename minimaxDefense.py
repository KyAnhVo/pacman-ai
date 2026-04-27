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
    FOOD_CLUSTER_WEIGHT_DIST = 2

    def __init__(self, layout: Layout, game_state: GameState, is_red: bool):
        self.layout = layout
        self.foods = set(
            (game_state.getRedFood() if is_red else game_state.getBlueFood()).asList()
        )
        self.entryPoints: set = self.calculate_entries()

    def group_foods(self):
        """
        Group foods together, close within FOOD_CLUSTER_WEIGHT_DIST
        """
        raise NotImplementedError()

    def calculate_entries(self):
        """
        Calculate the entry point set greedily.
        Greedy heuristic: choose the cell that forces the highest increase in average
        distance to the food groups
        """
        raise NotImplementedError()
