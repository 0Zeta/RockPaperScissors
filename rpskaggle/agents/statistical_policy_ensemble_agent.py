from random import randint
from rpskaggle.helpers import *
from rpskaggle.policies import *


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
        # Policies that shouldn´t receive a counter policy
        self.basic_policies = [RockPolicy(), PaperPolicy(), ScissorsPolicy()]
        self.advanced_policies = [FrequencyPolicy(), CopyLastActionPolicy()]
        # Counters to the advanced policies => sicilian reasoning
        self.counter_policies = [CounterPolicy(policy) for policy in self.advanced_policies]
        self.policies = self.random_policies + self.basic_policies + self.advanced_policies + self.counter_policies
        self.name_to_policy = {policy.name: policy for policy in self.policies}

        # Create a data frame with the probabilities the policies returned for every step
        policy_names = [policy.name for policy in self.policies]
        self.performance = pd.DataFrame(columns=['step'] + policy_names)
        self.performance.set_index('step', inplace=True)

    def act(self) -> int:
        if len(self.history) > 0:
            # Update the historical performance for each policy
            self.update_policy_performance()

        # Get the new probabilities from every policy
        policy_probs = np.array([policy.probabilities(self.step, self.score, self.history) for policy in self.policies])

        if len(self.history) > 0:
            # Determine the performance scores of the policies and calculate their respective weights using softmax
            policy_scores = self.performance.sum(axis=0).to_numpy()
            policy_weights = np.exp(policy_scores) / sum(np.exp(policy_scores))
            # Calculate the resulting probabilities for the possible actions
            probs = np.sum(policy_weights.reshape((policy_weights.size, 1)) * policy_probs, axis=0)

        # Play randomly for the first 45 steps
        if self.step < 45:
            return randint(0, 2)
        return int(np.random.choice(range(SIGNS), p=probs))

    def update_policy_performance(self):
        # Determine the scores for the different actions (Win: 1, Tie: 0, Loss: -1)
        scores = [0, 0, 0]
        opponent_action = self.obs.lastOpponentAction
        scores[(opponent_action + 1) % 3] = 1
        scores[(opponent_action + 2) % 3] = -1

        for policy_name, policy in self.name_to_policy.items():
            # Calculate the policy´s score for the last step
            probs = policy.history[-1]
            score = np.sum(probs * scores)
            # Save the score to the performance data frame
            self.performance.loc[self.step - 1, policy_name] = score


AGENT = None


def statistical_policy_ensemble(observation, configuration) -> int:
    global AGENT
    if AGENT is None:
        AGENT = StatisticalPolicyEnsembleAgent(configuration)
    action, history = AGENT.agent(observation)
    return action
