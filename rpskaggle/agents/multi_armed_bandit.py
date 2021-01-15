from random import randint
from rpskaggle.policies import *

logging.basicConfig(level=logging.INFO)


class MultiArmedBandit(RPSAgent):
    """
    a simple multi armed bandit approach sampling from a beta distribution
    """

    def __init__(self, configuration):
        super().__init__(configuration)
        # Load some policies
        self.policies = get_policies()
        self.name_to_policy = {policy.name: policy for policy in self.policies}
        policy_names = [policy.name for policy in self.policies]

        # Use one fixed decay value  TODO: use multiple different decays, points per win/loss and reset probabilities
        self.decays = [0.97]
        self.scores_by_decay = np.full(
            (len(self.decays), len(self.policies), 2), fill_value=2, dtype=np.float
        )
        # Award 3 points per win, -3 per loss and 0 for draws
        self.win = 3.0
        self.tie = 0.0
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
        self.win_loss_columns = [
            _
            for p in [[p_name + "_wins", p_name + "_losses"] for p_name in policy_names]
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
            # Determine the performance scores of the policies for each decay value, sample from the according beta
            # distribution and choose the policy with the highest score for each decay
            decay_probs = []
            for decay_index, decay in enumerate(self.decays):
                # Sample a value from the according beta distribution for every policy
                values = np.random.beta(
                    self.scores_by_decay[decay_index, :, 0],
                    self.scores_by_decay[decay_index, :, 1],
                )
                # Combine the probabilities of the policies with the three highest values
                highest = (-values).argsort()[:3]
                probs = 0.6 * policy_probs[highest[0]] + 0.25 * policy_probs[highest[1]] + 0.15 * policy_probs[highest[2]]
                decay_probs.append(probs)

            # TODO: implement multiple decay values
            probabilities = decay_probs[0]

            logging.info(
                "Multi Armed Bandit | Step "
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
        action = int(np.random.choice(range(SIGNS), p=probabilities))
        if get_score(self.history, 15) < -5:
            # If we got outplayed in the last 15 steps, play the counter of the chosen action´s counter with a
            # certain probability
            if randint(0, 100) <= 40:
                action = (action + 2) % SIGNS
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
            self.policies_performance.loc[self.step - 1, policy_name + "_wins"] = probs[
                winning_action
            ] * self.win + (probs[opponent_action] * self.tie if self.tie > 0 else 0)
            self.policies_performance.loc[
                self.step - 1, policy_name + "_losses"
            ] = probs[losing_action] * self.loss - (
                probs[opponent_action] * self.tie if self.tie < 0 else 0
            )
            # Reset after a loss
            if probs[losing_action] > 0.4:
                if np.random.random() < self.reset_prob:
                    self.scores_by_decay[
                        self.scores_by_decay[:, policy_index, 0]
                        > self.scores_by_decay[:, policy_index, 1],
                        policy_index,
                    ] = 2

        # Apply the different decay values
        for decay_index, decay in enumerate(self.decays):
            # Apply the decay
            self.scores_by_decay[decay_index] *= decay
            # Add the new scores
            self.scores_by_decay[decay_index, :, :] += (
                self.policies_performance.loc[self.step - 1, self.win_loss_columns]
                .to_numpy()
                .reshape(len(self.policies), 2)
                .astype(np.float)
            )


AGENT = None


def multi_armed_bandit(observation, configuration) -> int:
    global AGENT
    if AGENT is None:
        AGENT = MultiArmedBandit(configuration)
    action, history = AGENT.agent(observation)
    return action
