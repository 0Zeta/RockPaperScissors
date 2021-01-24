import logging
from collections import defaultdict
from typing import List
from sklearn.ensemble import RandomForestClassifier

from rpskaggle.agents.geometry_agent import GeometryPolicy
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
        if len(self.policy.history) == 0:
            return EQUAL_PROBS
        probs = self.policy.history[-1]
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
        self.name = sequence_name + "_sequence_policy"
        self.is_deterministic = True

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        return one_hot(self.sequence[step] % 3)


class DeterministicRPSContestPolicy(Policy):
    """
    a wrapper to run RPS Contest bots
    Adapted from https://www.kaggle.com/purplepuppy/running-rpscontest-bots
    """

    def __init__(self, code, agent_name):
        super().__init__()
        self.name = "deterministic_" + agent_name + "_policy"
        self.is_deterministic = True
        self.code = compile(code, "<string>", "exec")
        self.gg = dict()
        self.symbols = {"R": 0, "P": 1, "S": 2}

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        try:
            inp = (
                ""
                if len(history) < 1
                else "RPS"[int(history.loc[step - 1, "opponent_action"])]
            )
            out = (
                "" if len(history) < 1 else "RPS"[int(history.loc[step - 1, "action"])]
            )
            self.gg["input"] = inp
            self.gg["output"] = out
            exec(self.code, self.gg)
            return one_hot(self.symbols[self.gg["output"]])
        except Exception as exception:
            logging.error("An error ocurred in " + self.name + " : " + str(exception))
            return EQUAL_PROBS


class ProbabilisticRPSContestPolicy(Policy):
    """
    a wrapper to run modified probabilistic RPS Contest bots
    Adapted from https://www.kaggle.com/purplepuppy/running-rpscontest-bots
    """

    def __init__(self, code, agent_name):
        super().__init__()
        self.name = "probabilistic_" + agent_name + "_policy"
        self.is_deterministic = True
        self.code = compile(code, "<string>", "exec")
        self.gg = dict()
        self.symbols = {"R": 0, "P": 1, "S": 2}

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        try:
            inp = (
                ""
                if len(history) < 1
                else "RPS"[int(history.loc[step - 1, "opponent_action"])]
            )
            out = (
                "" if len(history) < 1 else "RPS"[int(history.loc[step - 1, "action"])]
            )
            self.gg["input"] = inp
            self.gg["output"] = out
            exec(self.code, self.gg)
            return np.array(self.gg["output"])
        except Exception as exception:
            logging.error("An error ocurred in " + self.name + " : " + str(exception))
            return EQUAL_PROBS


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
    prefers the longest matching sequence
    """

    def __init__(self, max_sequence_length: int):
        super().__init__()
        self.name = "max_history_policy"
        self.max_sequence_length = max_sequence_length
        self.sequences = defaultdict(lambda: np.zeros((3,), dtype=np.int))

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) < 2:
            # return equal probabilities at the start of the game
            return EQUAL_PROBS
        # Update the stored sequences with the opponent´s last move
        for sequence_length in range(
            1, min(len(history) - 1, self.max_sequence_length) + 1
        ):
            sequence = np.array2string(
                history.iloc[-sequence_length - 1 : -1].to_numpy()
            )
            self.sequences[sequence][int(history.loc[step - 1, "opponent_action"])] += 1
        # Try to find a match for the current history and get the corresponding probabilities
        for sequence_length in range(
            min(len(history), self.max_sequence_length), 0, -1
        ):
            # Determine whether the sequence has already occurred
            sequence = np.array2string(history.iloc[-sequence_length:].to_numpy())
            if sequence not in self.sequences.keys():
                continue
            # Return the corresponding probabilities
            return self.sequences[sequence] / sum(self.sequences[sequence])
        return EQUAL_PROBS


class MaxOpponentHistoryPolicy(Policy):
    """
    like MaxHistoryPolicy, but only looks at the moves of the opponent
    """

    def __init__(self, max_sequence_length: int):
        super().__init__()
        self.name = "max_opponent_history_policy"
        self.max_sequence_length = max_sequence_length
        self.sequences = defaultdict(lambda: np.zeros((3,), dtype=np.int))

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) < 2:
            # return equal probabilities at the start of the game
            return EQUAL_PROBS
        # Update the stored sequences with the opponent´s last move
        for sequence_length in range(
            1, min(len(history) - 1, self.max_sequence_length) + 1
        ):
            sequence = np.array2string(
                history.iloc[-sequence_length - 1 : -1][["opponent_action"]].to_numpy()
            )
            self.sequences[sequence][int(history.loc[step - 1, "opponent_action"])] += 1
        # Try to find a match for the current history and get the corresponding probabilities
        for sequence_length in range(
            min(len(history), self.max_sequence_length), 0, -1
        ):
            # Determine whether the sequence has already occurred
            sequence = np.array2string(
                history.iloc[-sequence_length:][["opponent_action"]].to_numpy()
            )
            if sequence not in self.sequences.keys():
                continue
            # Return the corresponding probabilities
            return self.sequences[sequence] / sum(self.sequences[sequence])
        return EQUAL_PROBS


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


class WinTieLosePolicy(Policy):
    """
    chooses the next move based on the result of the last one, e.g. repeats winning moves and switches when losing
    """

    def __init__(self, on_win: int, on_tie: int, on_lose: int):
        super().__init__()
        self.name = (
            "on_win_"
            + str(on_win)
            + "_on_tie_"
            + str(on_tie)
            + "_on_lose_"
            + str(on_lose)
            + "_policy"
        )
        self.is_deterministic = True
        self.on_win = on_win
        self.on_tie = on_tie
        self.on_lose = on_lose
        self.last_score = 0

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        if len(history) < 1:
            # Return equal probabilities on the first step
            return EQUAL_PROBS
        result = score - self.last_score
        self.last_score = score
        if result == 1:
            shift = self.on_win
        elif result == 0:
            shift = self.on_tie
        else:
            shift = self.on_lose
        return one_hot((int(history.loc[step - 1, "action"]) + shift) % 3)


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


def get_policies():
    """
    Returns a list of many polices
    """
    # Initialize the different sets of policies
    random_policies = [RandomPolicy()]
    # Policies we shouldn´t derive incremented ones from
    basic_policies = [RockPolicy(), PaperPolicy(), ScissorsPolicy()]
    advanced_policies = [
        FrequencyPolicy(),
        CopyLastActionPolicy(),
        TransitionMatrixPolicy(),
        TransitionTensorPolicy(),
        RandomForestPolicy(20, 20, 5),
        MaxHistoryPolicy(15),
        MaxOpponentHistoryPolicy(15),
        GeometryPolicy(),
        GeometryPolicy(0.05)
    ]
    # Add some popular sequences
    for seq_name, seq in SEQUENCES.items():
        advanced_policies.append(SequencePolicy(seq, seq_name))
    # Add some RPS Contest bots to the ensemble
    for agent_name, code in RPSCONTEST_BOTS.items():
        advanced_policies.append(DeterministicRPSContestPolicy(code, agent_name))
    # Strict versions of the advanced policies
    strict_policies = [
        StrictPolicy(policy)
        for policy in advanced_policies
        if not policy.is_deterministic
    ]
    # Counter policies
    counter_policies = [
        CounterPolicy(FrequencyPolicy()),
        CounterPolicy(CopyLastActionPolicy()),
        CounterPolicy(TransitionMatrixPolicy()),
        CounterPolicy(TransitionTensorPolicy()),
        CounterPolicy(MaxHistoryPolicy(15)),
        CounterPolicy(MaxOpponentHistoryPolicy(15)),
        CounterPolicy(WinTieLosePolicy(0, 1, 1)),
        CounterPolicy(WinTieLosePolicy(0, 2, 2)),
        CounterPolicy(GeometryPolicy())
    ]
    # Add some RPS Contest bots to the ensemble
    for agent_name, code in RPSCONTEST_BOTS.items():
        counter_policies.append(
            CounterPolicy(DeterministicRPSContestPolicy(code, agent_name))
        )
    strict_counter_policies = [
        StrictPolicy(policy)
        for policy in counter_policies
        if not policy.is_deterministic
    ]
    # Sicilian reasoning
    incremented_policies = [
        IncrementPolicy(policy)
        for policy in (
                advanced_policies
                + strict_policies
                + counter_policies
                + strict_counter_policies
        )
    ]
    double_incremented_policies = [
        IncrementPolicy(policy) for policy in incremented_policies
    ]
    policies = (
            random_policies
            + basic_policies
            + advanced_policies
            + strict_policies
            + incremented_policies
            + double_incremented_policies
            + counter_policies
            + strict_counter_policies
    )
    return policies
