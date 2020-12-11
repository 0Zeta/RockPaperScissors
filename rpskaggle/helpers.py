from typing import Tuple

import numpy as np
import pandas as pd


SIGNS = 3
EQUAL_PROBS = np.array([1, 1, 1], dtype=np.float) / 3


class RPSAgent(object):

    def __init__(self, configuration):
        self.obs = None
        self.config = configuration
        self.history = pd.DataFrame(columns=['step', 'action', 'opponent_action'])
        self.history.set_index('step', inplace=True)

        self.step = 0
        self.score = 0

    def agent(self, observation, configuration=None, history=None) -> Tuple[int, pd.DataFrame]:
        if configuration is not None:
            self.config = configuration
        if history is not None:
            self.history = history
        self.obs = observation

        self.step = self.obs.step

        # Append the last action of the opponent to the history
        if self.step > 0:
            self.history.loc[self.step - 1, 'opponent_action'] = self.obs.lastOpponentAction
            self.score = get_score(self.history)

        # Choose an action and append it to the history
        action = self.act()
        self.history.loc[self.step] = {'action': action, 'opponent_action': None}
        return action, self.history

    def act(self) -> int:
        pass


class Policy(object):

    def get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        """
        Returns probabilities for all possible actions
        """
        pass


def counters(actions: pd.Series) -> pd.Series:
    """
    Returns the counters for the specified actions
    """
    return (actions + 1) % SIGNS


def get_score(history: pd.DataFrame) -> int:
    score = 0
    score += len(history[((history['opponent_action'] + 1) % SIGNS) == history['action']])
    score -= len(history[((history['action'] + 1) % SIGNS) == history['opponent_action']])
    return score
