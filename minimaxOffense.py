from __future__ import annotations

from capture import SIGHT_RANGE, SONAR_NOISE_RANGE, GameState
from captureAgents import CaptureAgent
from game import Actions, Directions, Grid
from layout import Layout
from util import manhattanDistance


class EnemyBeliefTracker:
    """
    Tracks the belief distribution of enemy positions based on sonar noise.
    This is one pair of (ally, enemy), so O(n^2) pairs for n agents each team.

    Note: this belief tracker works the best when allies are further away from each other, thus the
    intersected area is small
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

        # If we see the enemy, return immediately
        enemy_pos = gameState.getAgentPosition(self.enemy_index)
        if enemy_pos:
            self.enemy_position_distribution = {enemy_pos}
            return

        agent_distances = gameState.getAgentDistances()
        if agent_distances is None:
            return

        enemy_distance = agent_distances[self.enemy_index]
        new_belief: set = set()
        for x, y in self.enemy_position_distribution:
            # 1: stretch distribution top, down, left right
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            for nx, ny in neighbors:
                if nx < 0 or nx > self.max_x:
                    continue
                if ny < 0 or ny > self.max_y:
                    continue
                if self.layout.isWall((nx, ny)):
                    continue

                # 2: check if the position is within the defined ring
                half_noise = SONAR_NOISE_RANGE // 2
                dist = manhattanDistance(
                    gameState.getAgentPosition(self.ally_index), (nx, ny)
                )
                if (
                    dist < enemy_distance - half_noise
                    or dist > enemy_distance + half_noise
                ):
                    continue

                new_belief.add((nx, ny))

        self.enemy_position_distribution = new_belief

    def update_with_ally(self, ally: EnemyBeliefTracker):
        """
        Intersect our enemy position belief with ally enemy position belief
        Only need to call this once per pair, as this syncs to the other agent also.
        """
        new_belief = self.enemy_position_distribution.intersection(
            ally.enemy_position_distribution
        )
        self.enemy_position_distribution = new_belief
        ally.enemy_position_distribution = new_belief


class GreedyThiefAgent(CaptureAgent):
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

    def registerInitialState(self, gameState: GameState):
        CaptureAgent.registerInitialState(self, gameState)

        # Get the walls for traversal
        self.walls = gameState.getWalls()

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
