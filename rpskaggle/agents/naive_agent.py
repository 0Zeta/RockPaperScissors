from random import randint
from rpskaggle.helpers import *


class NaiveAgent(RPSAgent):
    """
    A simple agent that plays randomly for the first steps and uses the frequencies of the opponent's actions to
    choose actions afterwards. When this doesn't work well, it switches to random play again.
    """

    def __init__(self, configuration):
        super().__init__(configuration)
        self.play_random = True

    def act(self) -> int:
        if self.should_choose_random():
            return randint(0, 2)

        # Use the relative frequencies of the opponent's actions as a probability distribution for our actions
        probs = counters(self.history["opponent_action"]).value_counts(
            normalize=True, sort=False
        )
        for i in range(SIGNS):
            if i not in probs.keys():
                probs.loc[i] = 0.0
        probs.sort_index(inplace=True)
        return int(np.random.choice(range(SIGNS), p=probs))

    def should_choose_random(self) -> bool:
        # Play random for the first 50 steps
        if self.step < 50:
            self.play_random = True
            return True
        # and don't play random for the following 50 steps
        elif self.step < 100:
            self.play_random = False
            return False

        if not self.play_random:
            if self.score < -20:
                # Seems like our strategy doesn't work
                self.play_random = True

        # Flip the decision with a certain probability
        return self.play_random if randint(1, 100) <= 85 else not self.play_random


AGENT = None


def naive_agent(observation, configuration) -> int:
    global AGENT
    if AGENT is None:
        AGENT = NaiveAgent(configuration)
    action, history = AGENT.agent(observation)
    return action
