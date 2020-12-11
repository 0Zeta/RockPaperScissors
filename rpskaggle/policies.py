from rpskaggle.helpers import *


class RandomPolicy(Policy):
    """
    returns equal probabilities for all actions
    """

    def get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return EQUAL_PROBS


class FrequencyPolicy(Policy):
    """
    chooses actions based on the frequency of the opponent´s last actions
    """

    def get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) == 0:
            # Return equal probabilities at the start of the episode
            return EQUAL_PROBS
        probs = counters(history['opponent_action']).value_counts(normalize=True, sort=False)
        for i in range(SIGNS):
            if i not in probs.keys():
                probs.loc[i] = 0.
        probs.sort_index(inplace=True)
        return probs.to_numpy()


class CopyLastActionPolicy(Policy):
    """
    copies the last action of the opponent
    """

    def get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) == 0:
            # Return equal probabilities at the start of the episode
            return EQUAL_PROBS
        probs = np.zeros((3,), dtype=np.float)
        probs[int(history.loc[step - 1, 'opponent_action'])] = 1.
        return probs
