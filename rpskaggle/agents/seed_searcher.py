# Importing important imports
import numpy as np
import pandas as pd
import random

# Global Variables
from rpskaggle.helpers import Policy, one_hot, EQUAL_PROBS


class SeedSearchPolicy(Policy):
    """
    Trying to crack seeds
    Adapted from Taaha Khan´s notebook "RPS: Cracking Random Number Generators"
    https://www.kaggle.com/taahakhan/rps-cracking-random-number-generators
    """

    def __init__(self, seed_count: int):
        super().__init__()
        self.name = "seed_search_policy"
        self.is_deterministic = True  # Actually not, but we don´t need a strict version
        self.seeds = list(range(seed_count))
        self.previous_moves = []

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        # Saving the current state
        init_state = random.getstate()
        next_move = -1
        # If there still are multiple candidates
        if len(history) > 0 and len(self.seeds) > 1:
            # Saving previous moves
            self.previous_moves.append(int(history.loc[step - 1, "opponent_action"]))
            # Checking each possible seed
            for i in range(len(self.seeds) - 1, -1, -1):
                # Running for previous moves
                random.seed(self.seeds[i])
                for s in range(step):
                    move = random.randint(0, 2)
                    # Testing their move order
                    if move != self.previous_moves[s]:
                        self.seeds.pop(i)
                        break
        # Seed found: Get the next move
        elif len(self.seeds) == 1:
            random.seed(self.seeds[0])
            for _ in range(step):
                move = random.randint(0, 2)
            next_move = random.randint(0, 2)

        # Resetting the state to not interfere with the opponent
        random.setstate(init_state)
        if next_move > -1:
            return one_hot((next_move + 1) % 3)
        return EQUAL_PROBS
