from typing import List
from sklearn.ensemble import RandomForestClassifier

from rpskaggle.helpers import *


class RandomPolicy(Policy):
    """
    returns equal probabilities for all actions
    """

    def __init__(self):
        super().__init__()
        self.name = "random_policy"

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return EQUAL_PROBS


class IncrementPolicy(Policy):
    """
    turns all actions given by a policy into their counters
    """

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        self.name = "incremented_" + policy.name
        self.is_deterministic = policy.is_deterministic

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
        self.name = "counter_" + policy.name
        self.is_deterministic = policy.is_deterministic

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        probs = self.policy._get_probs(
            step,
            -score,
            history.rename(
                columns={"action": "opponent_action", "opponent_action": "action"}
            ),
        )
        return np.roll(probs, 1)


class StrictPolicy(Policy):
    """
    always selects the action with the highest probability for a given policy
    """

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        self.name = "strict_" + policy.name
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        probs = self.policy._get_probs(step, score, history).copy()
        action = np.argmax(probs)
        probs[:] = 0
        probs[action] = 1
        return probs


class AlternatePolicy(Policy):
    """
    Alternates between the specified policies every interval steps
    """

    def __init__(self, policies: List[Policy], interval: int):
        super().__init__()
        self.name = (
            "alternate_"
            + ("_".join([policy.name.replace("_policy", "") for policy in policies]))
            + "_policies"
        )
        self.is_deterministic = all([policy.is_deterministic for policy in policies])
        self.policies = policies
        self.interval = interval
        self.current_policy = 0

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if step % self.interval == 0:
            # Alternate
            self.current_policy = (self.current_policy + 1) % len(self.policies)
        return self.policies[self.current_policy]._get_probs(step, score, history)


class SequencePolicy(Policy):
    """
    chooses actions from a specified sequence
    """

    def __init__(self, sequence: List[int], sequence_name: str):
        super().__init__()
        self.sequence = sequence
        self.name = sequence_name + '_sequence_policy'
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return one_hot(self.sequence[step] % 3)


class FrequencyPolicy(Policy):
    """
    chooses actions based on the frequency of the opponent´s last actions
    """

    def __init__(self):
        super().__init__()
        self.name = "frequency_policy"

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) == 0:
            # Return equal probabilities at the start of the episode
            return EQUAL_PROBS
        probs = counters(history["opponent_action"]).value_counts(
            normalize=True, sort=False
        )
        for i in range(SIGNS):
            if i not in probs.keys():
                probs.loc[i] = 0.0
        probs.sort_index(inplace=True)
        return probs.to_numpy()


class CopyLastActionPolicy(Policy):
    """
    copies the last action of the opponent
    """

    def __init__(self):
        super().__init__()
        self.name = "copy_last_action_policy"
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) == 0:
            # Return equal probabilities at the start of the episode
            return EQUAL_PROBS
        probs = np.zeros((3,), dtype=np.float)
        probs[int(history.loc[step - 1, "opponent_action"])] = 1.0
        return probs


class TransitionMatrixPolicy(Policy):
    """
    uses a simple transition matrix to predict the opponent´s next action and counter it

    Adapted from https://www.kaggle.com/group16/rps-opponent-transition-matrix
    """

    def __init__(self):
        super().__init__()
        self.name = "transition_matrix_policy"
        self.T = np.zeros((3, 3), dtype=np.int)
        self.P = np.zeros((3, 3), dtype=np.float)

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) > 1:
            # Update the matrices
            last_action = int(history.loc[step - 1, "opponent_action"])
            self.T[int(history.loc[step - 2, "opponent_action"]), last_action] += 1
            self.P = np.divide(self.T, np.maximum(1, self.T.sum(axis=1)).reshape(-1, 1))
            if np.sum(self.P[last_action, :]) == 1:
                return np.roll(self.P[last_action, :], 1)
        return EQUAL_PROBS


class TransitionTensorPolicy(Policy):
    """
    similar to TransitionMatrixPolicy, but takes both agent´s actions into account
    """

    def __init__(self):
        super().__init__()
        self.name = "transition_tensor_policy"
        self.T = np.zeros((3, 3, 3), dtype=np.int)
        self.P = np.zeros((3, 3, 3), dtype=np.float)

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) > 1:
            # Update the matrices
            last_action = int(history.loc[step - 1, "action"])
            opponent_last_action = int(history.loc[step - 1, "opponent_action"])
            self.T[
                int(history.loc[step - 2, "opponent_action"]),
                int(history.loc[step - 2, "action"]),
                last_action,
            ] += 1
            self.P = np.divide(
                self.T, np.maximum(1, self.T.sum(axis=2)).reshape(-1, 3, 1)
            )
            if np.sum(self.P[opponent_last_action, last_action, :]) == 1:
                return np.roll(self.P[opponent_last_action, last_action, :], 1)
        return EQUAL_PROBS


class MaxHistoryPolicy(Policy):
    """
    searches for similar situations in the game history and assumes the past is doomed to repeat itself
    """

    def __init(self):
        super().__init__()
        self.name = "max_history_policy"

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        # TODO: implement
        pass


class RandomForestPolicy(Policy):
    """
    uses a random forest classificator to predict the opponent´s action using the last moves as data
    """

    def __init__(self, n_estimators: int, max_train_size: int, prediction_window: int):
        super().__init__()
        self.name = "random_forest_policy"
        self.is_deterministic = True
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        self.max_train_size = max_train_size
        self.prediction_window = prediction_window
        self.X_train = np.ndarray(shape=(0, prediction_window * 2), dtype=np.int)
        self.y_train = np.ndarray(shape=(0, 1), dtype=np.int)

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) < self.prediction_window + 1:
            # Return equal probabilities until we have enough data
            return EQUAL_PROBS
        # Add the last prediction_window steps to the training data
        last_steps = history.iloc[-self.prediction_window - 1 : -1][
            ["action", "opponent_action"]
        ].to_numpy()
        self.X_train = np.append(
            self.X_train, last_steps.reshape(1, self.prediction_window * 2)
        )
        self.y_train = np.append(self.y_train, history.iloc[-1]["opponent_action"])
        self.X_train = self.X_train.reshape(-1, self.prediction_window * 2)
        # Ensure we don´t use more than max_train_size samples
        if len(self.X_train) > self.max_train_size:
            self.X_train = self.X_train[1:]
            self.y_train = self.y_train[1:]
        # Fit the model
        self.model.fit(self.X_train, self.y_train)
        # Predict the opponent´s next action
        X_predict = history.iloc[-self.prediction_window :][
            ["action", "opponent_action"]
        ].to_numpy()
        prediction = self.model.predict(
            X_predict.reshape(1, self.prediction_window * 2)
        )
        # Return the counter of the action
        return np.roll(one_hot(int(prediction[0])), 1)


class RockPolicy(Policy):
    """
    chooses Rock the whole time
    """

    def __init__(self):
        super().__init__()
        self.name = "rock_policy"
        self.probs = np.array([1, 0, 0], dtype=np.float)
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return self.probs


class PaperPolicy(Policy):
    """
    chooses Paper the whole time
    """

    def __init__(self):
        super().__init__()
        self.name = "paper_policy"
        self.probs = np.array([0, 1, 0], dtype=np.float)
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return self.probs


class ScissorsPolicy(Policy):
    """
    chooses Scissors the whole time
    """

    def __init__(self):
        super().__init__()
        self.name = "scissors_policy"
        self.probs = np.array([0, 0, 1], dtype=np.float)
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return self.probs
