from random import randint
from rpskaggle.policies import *

logging.basicConfig(level=logging.INFO)


class MultiArmedBandit(RPSAgent):
    """
    a simple multi armed bandit approach sampling from a dirichlet distribution
    """

    def __init__(self, configuration):
        super().__init__(configuration)
        # Load some policies
        self.policies = get_policies()
        self.name_to_policy = {policy.name: policy for policy in self.policies}
        policy_names = [policy.name for policy in self.policies]

        self.decays = [0.8, 0.93, 0.97, 0.99]
        self.scores_by_decay = np.full(
            (len(self.decays), len(self.policies), 3), fill_value=2, dtype=np.float
        )
        # Record the performance of the different decay values
        self.decay_performance = pd.DataFrame(
            columns=["step"]
            + [
                _
                for d in [
                    [str(decay) + "_wins", str(decay) + "_ties", str(decay) + "_losses"]
                    for decay in self.decays
                ]
                for _ in d
            ]
        )
        self.decay_performance.set_index("step", inplace=True)
        self.decay_probs = list()
        self.decay_performance_decay = 0.95
        # Award 3 points per win, -3 per loss and 0 for draws
        self.win = 3.0
        self.tie = 3.0
        self.loss = 3.0
        # Probability of a reset after a policy loses once
        self.reset_prob = 0.1
        # Record the performance of all policies
        self.policies_performance = pd.DataFrame(
            columns=["step"]
            + [
                _
                for p in [
                    [p_name + "_wins", p_name + "_ties", p_name + "_losses"]
                    for p_name in policy_names
                ]
                for _ in p
            ]
        )
        self.win_tie_loss_columns = [
            _
            for p in [
                [p_name + "_wins", p_name + "_ties", p_name + "_losses"]
                for p_name in policy_names
            ]
            for _ in p
        ]
        self.policies_performance.set_index("step", inplace=True)

    def act(self) -> int:
        if len(self.history) > 0:
            # Update the historical performance for each policy
            self.update_performance()

        # Get the new probabilities from every policy
        policy_probs = np.array(
            [
                policy.probabilities(self.step, self.score, self.history)
                for policy in self.policies
            ]
        )

        if len(self.history) > 0:
            # Determine the performance scores of the policies for each decay value, sample from the according dirichlet
            # distribution and choose the policy with the highest score for each decay
            decay_probs = []
            for decay_index, decay in enumerate(self.decays):
                # Sample a value from the according dirichlet distribution for every policy
                values = np.ndarray(
                    shape=(self.scores_by_decay.shape[1],), dtype=np.float
                )
                for i in range(self.scores_by_decay.shape[1]):
                    dirichlet = np.random.dirichlet(
                        [
                            self.scores_by_decay[decay_index, i, 0],
                            self.scores_by_decay[decay_index, i, 1],
                            self.scores_by_decay[decay_index, i, 2],
                        ]
                    )
                    values[i] = dirichlet[0] - dirichlet[2]

                # Combine the probabilities of the policies with the three highest values
                highest = (-values).argsort()[:3]
                logging.debug(self.policies[highest[0]].name)
                probs = (
                    0.6 * policy_probs[highest[0]]
                    + 0.25 * policy_probs[highest[1]]
                    + 0.15 * policy_probs[highest[2]]
                )
                decay_probs.append(probs)

            self.decay_probs.append(decay_probs)
            chosen_decay = self.decays[0]
            if len(self.decay_performance) == 0:
                probabilities = decay_probs[0]
            else:
                # Choose the decay with the highest value sampled from a dirichlet distribution
                values = np.ndarray(shape=(len(self.decays),), dtype=np.float)
                decayed_decay_performance = (
                    (
                        self.decay_performance
                        * np.flip(
                            np.power(
                                self.decay_performance_decay,
                                np.arange(len(self.decay_performance)),
                            )
                        ).reshape((-1, 1))
                    )
                    .sum(axis=0)
                    .to_numpy()
                    .reshape((-1, 3))
                )
                decayed_decay_performance = (
                    decayed_decay_performance - np.min(decayed_decay_performance) + 0.01
                )
                for decay_index, _ in enumerate(self.decays):
                    dirichlet = np.random.dirichlet(
                        [
                            3 * decayed_decay_performance[decay_index, 0],
                            3 * decayed_decay_performance[decay_index, 1],
                            3 * decayed_decay_performance[decay_index, 2],
                        ]
                    )
                    values[decay_index] = dirichlet[0] - dirichlet[2]
                chosen_decay_index = int(np.argmax(values))
                probabilities = decay_probs[chosen_decay_index]
                chosen_decay = self.decays[chosen_decay_index]

            logging.info(
                "Multi Armed Bandit | Step "
                + str(self.step)
                + " | score: "
                + str(self.score)
                + " decay: "
                + str(chosen_decay)
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
        action = int(np.random.choice(range(SIGNS), p=probabilities))
        return action

    def update_performance(self):
        opponent_action = self.obs.lastOpponentAction
        winning_action = (opponent_action + 1) % 3
        losing_action = (opponent_action + 2) % 3

        # Policies
        for policy_index, (policy_name, policy) in enumerate(
            self.name_to_policy.items()
        ):
            # Update the policy scores
            probs = policy.history[-1]
            self.policies_performance.loc[self.step - 1, policy_name + "_wins"] = (
                probs[winning_action] * self.win
            )
            self.policies_performance.loc[self.step - 1, policy_name + "_losses"] = (
                probs[losing_action] * self.loss
            )
            self.policies_performance.loc[self.step - 1, policy_name + "_ties"] = (
                probs[opponent_action] * self.tie
            )
            # Reset after a loss
            if probs[losing_action] > 0.4:
                if np.random.random() < self.reset_prob:
                    self.scores_by_decay[
                        self.scores_by_decay[:, policy_index, 0]
                        > self.scores_by_decay[:, policy_index, 2],
                        policy_index,
                    ] = 2

        # Decays
        if len(self.decay_probs) > 0:
            for decay_index, decay in enumerate(self.decays):
                probs = self.decay_probs[-1][decay_index]
                self.decay_performance.loc[self.step - 1, str(decay) + "_wins"] = probs[
                    winning_action
                ]
                self.decay_performance.loc[
                    self.step - 1, str(decay) + "_losses"
                ] = probs[losing_action]
                self.decay_performance.loc[self.step - 1, str(decay) + "_ties"] = probs[
                    opponent_action
                ]

        # Apply the different decay values
        for decay_index, decay in enumerate(self.decays):
            # Apply the decay
            self.scores_by_decay[decay_index] *= decay
            # Add the new scores
            self.scores_by_decay[decay_index, :, :] += (
                self.policies_performance.loc[self.step - 1, self.win_tie_loss_columns]
                .to_numpy()
                .reshape(len(self.policies), 3)
                .astype(np.float)
            )


AGENT = None


def multi_armed_bandit(observation, configuration) -> int:
    global AGENT
    if AGENT is None:
        AGENT = MultiArmedBandit(configuration)
    action, history = AGENT.agent(observation)
    return action
