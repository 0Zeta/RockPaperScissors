from rpskaggle.helpers import *


class SinglePolicyAgent(RPSAgent):
    """
    An agent employing a single policy
    """

    def __init__(self, policy: Policy, configuration):
        super().__init__(configuration)
        self.policy = policy

    def act(self) -> int:
        return int(np.random.choice(range(SIGNS), p=self.policy.probabilities(self.step, self.score, self.history)))
