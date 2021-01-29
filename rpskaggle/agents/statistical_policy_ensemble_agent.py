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
        self.policies = get_policies()
        if self.strict_agent:
            self.policies = [
                policy for policy in self.policies if policy.is_deterministic
            ]
        self.name_to_policy = {policy.name: policy for policy in self.policies}

        # The different combinations of decay values, reset probabilities and zero clips
        self.configurations = [
            (0.7, 0.0, False),
            (0.8, 0.0, False),
            (0.8866, 0.01, False),
            (0.8866, 0.0, False),
            (0.93, 0.05, False),
            (0.93, 0.0, False),
            (0.9762, 0.05, True),
            (0.9880, 0.0, False),
            (0.9880, 0.1, False),
            (0.9966, 0.1, True),
            (0.99815, 0.1, False),
            (0.9995, 0.1, True),
            (1.0, 0.1, True),
            (1.0, 0.0, False),
        ]

        self.configuration_performance_decay = 0.98

        # Create a data frame with the historical performance of the policies
        policy_names = [policy.name for policy in self.policies]
        self.policies_performance = pd.DataFrame(columns=["step"] + policy_names)
        self.policies_performance.set_index("step", inplace=True)

        # The last scores for each configuration
        self.policy_scores_by_configuration = np.zeros(
            (len(self.configurations), len(self.policies)), dtype=np.float64
        )

        # Also record the performance of the different configurations
        self.last_probabilities_by_configuration = {
            decay: EQUAL_PROBS for decay in self.configurations
        }
        self.configurations_performance = pd.DataFrame(
            columns=["step"] + [str(config) for config in self.configurations]
        )
        self.configurations_performance.set_index("step", inplace=True)

    def act(self) -> int:
        if len(self.history) > 0:
            # Update the historical performance for each policy and for each decay value
            self.update_performance()

        # Get the new probabilities from every policy
        policy_probs = np.array(
            [
                policy.probabilities(self.step, self.score, self.history)
                for policy in self.policies
            ]
        )

        if len(self.history) > 0:
            # Determine the performance scores of the policies for each configuration and calculate their respective weights using a dirichlet distribution
            config_probs = []
            for config_index, conf in enumerate(self.configurations):
                decay, reset_prob, clip_zero = conf
                policy_scores = self.policy_scores_by_configuration[config_index, :]
                scale = 5 / (np.sum(np.power(decay, np.arange(0, 12))))
                policy_weights = np.random.dirichlet(scale * (policy_scores - np.min(policy_scores)) + 0.1)
                # Calculate the resulting probabilities for the possible actions
                p = np.sum(
                    policy_weights.reshape((policy_weights.size, 1)) * policy_probs,
                    axis=0,
                )
                highest = (-policy_weights).argsort()[:3]
                p = 0.7 * policy_probs[highest[0]] + 0.2 * policy_probs[highest[1]] + 0.1 * policy_probs[highest[2]]
                if self.strict_agent:
                    p = one_hot(int(np.argmax(p)))
                config_probs.append(p)
                # Save the probabilities to evaluate the performance of this decay value in the next step
                self.last_probabilities_by_configuration[conf] = p
                logging.debug(
                    "Configuration " + str(conf) + " probabilities: " + str(p)
                )

            # Determine the performance scores for the different configurations and calculate their respective weights
            # Apply a decay to the historical scores
            configuration_scores = (
                self.configurations_performance
                * np.flip(
                    np.power(
                        self.configuration_performance_decay,
                        np.arange(len(self.configurations_performance)),
                    )
                ).reshape((-1, 1))
            ).sum(axis=0) * 3
            configuration_weights = np.random.dirichlet(
                configuration_scores - np.min(configuration_scores) + 0.01
            )
            for decay_index, probs in enumerate(config_probs):
                if np.min(probs) > 0.25:
                    # Don't take predictions with a high amount of uncertainty into account
                    configuration_weights[decay_index] = 0
            if np.sum(configuration_weights) > 0.2:
                configuration_weights *= 1 / np.sum(configuration_weights)
                # Select the configuration with the highest value
                probabilities = config_probs[np.argmax(configuration_weights)]
            else:
                probabilities = EQUAL_PROBS
            logging.info(
                "Statistical Policy Ensemble | Step "
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

        #  Configurations
        for config_index, config in enumerate(self.configurations):
            decay, reset_prob, clip_zero = config
            # Calculate the score for the last step
            probs = self.last_probabilities_by_configuration[config]
            score = np.sum(probs * scores)
            # Apply the decay to the current score and add the new scores
            new_scores = (
                self.policy_scores_by_configuration[config_index] * decay
                + self.policies_performance.loc[self.step - 1, :].to_numpy()
            )
            # Zero clip
            if clip_zero:
                new_scores[new_scores < 0] = 0
            # Reset losing policies with a certain probability
            if reset_prob > 0:
                policy_scores = self.policies_performance.loc[
                    self.step - 1, :
                ].to_numpy()
                to_reset = np.logical_and(
                    policy_scores < -0.4,
                    new_scores > 0,
                    np.random.random(len(self.policies)) < reset_prob,
                )
                new_scores[to_reset] = 0
            self.policy_scores_by_configuration[config_index] = new_scores
            # Save the score to the performance data frame
            self.configurations_performance.loc[self.step - 1, str(config)] = score


AGENT = None


def statistical_policy_ensemble(observation, configuration) -> int:
    global AGENT
    if AGENT is None:
        AGENT = StatisticalPolicyEnsembleAgent(configuration)
    action, history = AGENT.agent(observation)
    return action
