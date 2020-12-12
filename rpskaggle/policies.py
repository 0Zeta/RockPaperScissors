from rpskaggle.helpers import *


class RandomPolicy(Policy):
    """
    returns equal probabilities for all actions
    """

    def __init__(self):
        super().__init__()
        self.name = 'random_policy'

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return EQUAL_PROBS


class IncrementPolicy(Policy):
    """
    turns all actions given by a policy into their counters
    """

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        self.name = 'counter_' + policy.name

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        # Return equal probabilities if the history of the policy is empty
        if len(self.policy.history) == 0:
            return EQUAL_PROBS
        return np.roll(self.policy.history[-1], 1)


class CounterPolicy(Policy):
    """
    a policy countering the specified policy assuming the opponent uses this policy
    """

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        self.name = 'opponent_' + policy.name

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        probs = self.policy._get_probs(step, -score, history.rename(columns={'action': 'opponent_action', 'opponent_action': 'action'}))
        return np.roll(probs, 1)


class StrictPolicy(Policy):
    """
    always selects the action with the highest probability for a given policy
    """

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        self.name = 'strict_' + policy.name

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        probs = self.policy._get_probs(step, score, history)
        action = np.argmax(probs)
        probs[:] = 0
        probs[action] = 1
        return probs


class FrequencyPolicy(Policy):
    """
    chooses actions based on the frequency of the opponent´s last actions
    """

    def __init__(self):
        super().__init__()
        self.name = 'frequency_policy'

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
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

    def __init__(self):
        super().__init__()
        self.name = 'copy_last_action_policy'

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) == 0:
            # Return equal probabilities at the start of the episode
            return EQUAL_PROBS
        probs = np.zeros((3,), dtype=np.float)
        probs[int(history.loc[step - 1, 'opponent_action'])] = 1.
        return probs


class RockPolicy(Policy):
    """
    chooses Rock the whole time
    """

    def __init__(self):
        super().__init__()
        self.name = 'rock_policy'
        self.probs = np.array([1, 0, 0], dtype=np.float)

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return self.probs


class PaperPolicy(Policy):
    """
    chooses Paper the whole time
    """

    def __init__(self):
        super().__init__()
        self.name = 'paper_policy'
        self.probs = np.array([0, 1, 0], dtype=np.float)

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return self.probs


class ScissorsPolicy(Policy):
    """
    chooses Scissors the whole time
    """

    def __init__(self):
        super().__init__()
        self.name = 'scissors_policy'
        self.probs = np.array([0, 0, 1], dtype=np.float)

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return self.probs
