import logging
from random import randint
from rpskaggle.policies import *


logging.basicConfig(level=logging.INFO)


class StatisticalPolicyEnsembleAgent(RPSAgent):
    """
    evaluates the performance of different policies and assigns each policy a weight based on the policy´s
    historical performance
    After that the combined weighted probabilities from the policies are used as a probability distribution
    for the agent´s actions
    """

    def __init__(self, configuration):
        super().__init__(configuration)

        # Initialize the different sets of policies
        self.random_policies = [RandomPolicy()]
        # Policies we shouldn´t derive incremented ones from
        self.basic_policies = [RockPolicy(), PaperPolicy(), ScissorsPolicy()]
        self.advanced_policies = [
            FrequencyPolicy(),
            CopyLastActionPolicy(),
            TransitionMatrixPolicy(),
            TransitionTensorPolicy(),
            RandomForestPolicy(20, 20, 5),
            MaxHistoryPolicy(15),
            MaxOpponentHistoryPolicy(15),
        ]
        # Add some popular sequences
        for seq_name, seq in SEQUENCES.items():
            self.advanced_policies.append(SequencePolicy(seq, seq_name))
        # Strict versions of the advanced policies
        self.strict_policies = [
            StrictPolicy(policy)
            for policy in self.advanced_policies
            if not policy.is_deterministic
        ]
        # Counter policies
        self.counter_policies = [
            CounterPolicy(FrequencyPolicy()),
            CounterPolicy(CopyLastActionPolicy()),
            CounterPolicy(TransitionMatrixPolicy()),
            CounterPolicy(TransitionTensorPolicy()),
            CounterPolicy(MaxHistoryPolicy(15)),
            CounterPolicy(MaxOpponentHistoryPolicy(15)),
            CounterPolicy(WinTieLosePolicy(0, 1, 1)),
            CounterPolicy(WinTieLosePolicy(0, 2, 2)),
        ]
        self.strict_counter_policies = [
            StrictPolicy(policy)
            for policy in self.counter_policies
            if not policy.is_deterministic
        ]
        # Sicilian reasoning
        self.incremented_policies = [
            IncrementPolicy(policy)
            for policy in (
                self.advanced_policies
                + self.strict_policies
                + self.counter_policies
                + self.strict_counter_policies
            )
        ]
        self.double_incremented_policies = [
            IncrementPolicy(policy) for policy in self.incremented_policies
        ]
        self.policies = (
            self.random_policies
            + self.advanced_policies
            + self.strict_policies
            + self.incremented_policies
            + self.double_incremented_policies
            + self.counter_policies
            + self.strict_counter_policies
        )
        self.name_to_policy = {policy.name: policy for policy in self.policies}

        # The different options for how many of the last steps are taken into account when calculating a policy´s performance
        self.window_sizes = [5, 10, 50, 100, 350, 650, 1000]

        # Create a data frame with the historical performance of the policies
        policy_names = [policy.name for policy in self.policies]
        self.policies_performance = pd.DataFrame(columns=["step"] + policy_names)
        self.policies_performance.set_index("step", inplace=True)

        # Also record the performance of the different performance window sizes
        self.last_probabilities_by_window_size = {
            window_size: EQUAL_PROBS for window_size in self.window_sizes
        }
        self.window_sizes_performance = pd.DataFrame(
            columns=["step"] + [str(size) for size in self.window_sizes]
        )
        self.window_sizes_performance.set_index("step", inplace=True)

    def act(self) -> int:
        logging.debug("Begin step " + str(self.step))
        if len(self.history) > 0:
            # Update the historical performance for each policy and for each window size
            self.update_performance()

        # Get the new probabilities from every policy
        policy_probs = np.array(
            [
                policy.probabilities(self.step, self.score, self.history)
                for policy in self.policies
            ]
        )

        if len(self.history) > 0:
            # Determine the performance scores of the policies for each window size and calculate their respective weights using softmax
            window_probs = []
            for window_size in self.window_sizes:
                policy_scores = self.policies_performance.tail(window_size).sum(axis=0)
                if window_size == self.window_sizes[-1]:
                    logging.debug(policy_scores)
                policy_scores = policy_scores.to_numpy() / max(window_size / 200, 2)
                policy_weights = np.exp(policy_scores - np.max(policy_scores)) / sum(
                    np.exp(policy_scores - np.max(policy_scores))
                )
                # Calculate the resulting probabilities for the possible actions
                p = np.sum(
                    policy_weights.reshape((policy_weights.size, 1)) * policy_probs,
                    axis=0,
                )
                window_probs.append(p)
                # Save the probabilities to evaluate the performance of this window size in the next step
                self.last_probabilities_by_window_size[window_size] = p
                logging.debug(
                    "Window size " + str(window_size) + " probabilities: " + str(p)
                )

            # Determine the performance scores for the window sizes and calculate their respective weights
            window_scores = self.window_sizes_performance.sum(axis=0)
            window_scores = window_scores.to_numpy() / 5
            window_weights = np.exp(window_scores - np.max(window_scores)) / sum(
                np.exp(window_scores - np.max(window_scores))
            )
            # Calculate the resulting probabilities for the possible actions
            probabilities = np.sum(
                window_weights.reshape((window_weights.size, 1)) * window_probs,
                axis=0,
            )
            logging.debug("Probabilities: " + str(probabilities))

        # Play randomly for the first 45 steps
        if self.step < 45:
            return randint(0, 2)
        return int(np.random.choice(range(SIGNS), p=probabilities))

    def update_performance(self):
        # Determine the scores for the different actions (Win: 1, Tie: 0, Loss: -1)
        scores = [0, 0, 0]
        opponent_action = self.obs.lastOpponentAction
        scores[(opponent_action + 1) % 3] = 1
        scores[(opponent_action + 2) % 3] = -1

        # Policies
        for policy_name, policy in self.name_to_policy.items():
            # Calculate the policy´s score for the last step
            probs = policy.history[-1]
            score = np.sum(probs * scores)
            # Save the score to the performance data frame
            self.policies_performance.loc[self.step - 1, policy_name] = score

        # Window sizes
        for window_size in self.window_sizes:
            # Calculate the score for the last step
            probs = self.last_probabilities_by_window_size[window_size]
            score = np.sum(probs * scores)
            # Save the score to the performance data frame
            self.window_sizes_performance.loc[self.step - 1, str(window_size)] = score


AGENT = None


def statistical_policy_ensemble(observation, configuration) -> int:
    global AGENT
    if AGENT is None:
        AGENT = StatisticalPolicyEnsembleAgent(configuration)
    action, history = AGENT.agent(observation)
    return action
