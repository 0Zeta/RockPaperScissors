import logging
import math
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

    def __init__(self, configuration, strict: bool = False):
        super().__init__(configuration)
        self.strict_agent = strict
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
        # Add some RPS Contest bots to the ensemble
        for agent_name, code in RPSCONTEST_BOTS.items():
            self.advanced_policies.append(RPSContestPolicy(code, agent_name))
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
        # Add some RPS Contest bots to the ensemble
        for agent_name, code in RPSCONTEST_BOTS.items():
            self.counter_policies.append(
                CounterPolicy(RPSContestPolicy(code, agent_name))
            )
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
        if self.strict_agent:
            self.policies = [
                policy for policy in self.policies if policy.is_deterministic
            ]
        self.name_to_policy = {policy.name: policy for policy in self.policies}

        # The different decay values
        self.decay_values = [0.7, 0.8866, 0.9762, 0.9880, 0.9966, 0.99815, 0.9995, 1.0]

        # Create a data frame with the historical performance of the policies
        policy_names = [policy.name for policy in self.policies]
        self.policies_performance = pd.DataFrame(columns=["step"] + policy_names)
        self.policies_performance.set_index("step", inplace=True)

        # The last scores for each decay value
        self.policy_scores_by_decay = np.zeros(
            (len(self.decay_values), len(self.policies)), dtype=np.float64
        )

        # Also record the performance of the different decay values
        self.last_probabilities_by_decay = {
            decay: EQUAL_PROBS for decay in self.decay_values
        }
        self.decays_performance = pd.DataFrame(
            columns=["step"] + [str(size) for size in self.decay_values]
        )
        self.decays_performance.set_index("step", inplace=True)

    def act(self) -> int:
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
            # Determine the performance scores of the policies for each decay value and calculate their respective weights using softmax
            decay_probs = []
            for decay_index, decay in enumerate(self.decay_values):
                policy_scores = self.policy_scores_by_decay[decay_index, :]
                policy_scores = policy_scores / min(
                    max(math.log(decay, 0.3) / 200, 1), 5
                )
                policy_weights = np.exp(policy_scores - np.max(policy_scores)) / sum(
                    np.exp(policy_scores - np.max(policy_scores))
                )
                # Calculate the resulting probabilities for the possible actions
                p = np.sum(
                    policy_weights.reshape((policy_weights.size, 1)) * policy_probs,
                    axis=0,
                )
                if self.strict_agent:
                    p = one_hot(int(np.argmax(p)))
                decay_probs.append(p)
                # Save the probabilities to evaluate the performance of this window size in the next step
                self.last_probabilities_by_decay[decay] = p
                logging.debug("Window size " + str(decay) + " probabilities: " + str(p))

            # Determine the performance scores for the window sizes and calculate their respective weights
            # Use the last 50 steps
            window_scores = self.decays_performance.sum(axis=0)
            window_scores = window_scores.to_numpy() / 5
            window_weights = np.exp(window_scores - np.max(window_scores)) / sum(
                np.exp(window_scores - np.max(window_scores))
            )
            # Calculate the resulting probabilities for the possible actions
            probabilities = np.sum(
                window_weights.reshape((window_weights.size, 1)) * decay_probs,
                axis=0,
            )
            logging.info(
                "Step "
                + str(self.step)
                + " | score: "
                + str(self.score)
                + " probabilities: "
                + str(probabilities)
            )

        # Play randomly for the first 100-200 steps
        if self.step < 100 + randint(0, 100):
            action = randint(0, 2)
            if randint(0, 3) == 1:
                # We don´t want our random seed to be cracked.
                action = (action + 1) % SIGNS
            return action
        if self.strict_agent:
            action = int(np.argmax(probabilities))
        else:
            action = int(np.random.choice(range(SIGNS), p=probabilities))
        if get_score(self.history, 15) < -5:
            # If we got outplayed in the last 15 steps, play the counter of the chosen action´s counter with a
            # certain probability
            if randint(0, 100) <= 40:
                action = (action + 2) % SIGNS
        return action

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

        # Decay values
        for decay_index, decay in enumerate(self.decay_values):
            # Calculate the score for the last step
            probs = self.last_probabilities_by_decay[decay]
            score = np.sum(probs * scores)
            # Apply the decay to the current score and add the new scores
            self.policy_scores_by_decay[decay_index] = (
                self.policy_scores_by_decay[decay_index] * decay
                + self.policies_performance.loc[self.step - 1, :].to_numpy()
            )
            # Save the score to the performance data frame
            self.decays_performance.loc[self.step - 1, str(decay)] = score


AGENT = None


def statistical_policy_ensemble(observation, configuration) -> int:
    global AGENT
    if AGENT is None:
        AGENT = StatisticalPolicyEnsembleAgent(configuration)
    action, history = AGENT.agent(observation)
    return action
