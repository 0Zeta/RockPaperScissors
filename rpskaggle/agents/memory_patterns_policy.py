"""
Memory Patterns agent by Yegor Biryukov
Adapted from https://www.kaggle.com/yegorbiryukov/rock-paper-scissors-with-memory-patterns
"""
import random
import numpy as np
import pandas as pd

from rpskaggle.helpers import Policy, one_hot

# maximum steps in a memory pattern
STEPS_MAX = 5
# minimum steps in a memory pattern
STEPS_MIN = 3
# lowest efficiency threshold of a memory pattern before being removed from agent's memory
EFFICIENCY_THRESHOLD = -3


class MemoryPatternsPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.name = "memory_patterns_policy"
        self.is_deterministic = True

        # current memory of the agent
        self.current_memory = []
        # previous action of my_agent
        self.previous_action = {
            "action": None,
            # action was taken from pattern
            "action_from_pattern": False,
            "pattern_group_index": None,
            "pattern_index": None,
        }
        # maximum length of current_memory
        self.current_memory_max_length = STEPS_MAX * 2
        # current reward of my_agent
        # will be taken from observation in the next release of kaggle environments
        self.reward = 0
        # memory length of patterns in first group
        # STEPS_MAX is multiplied by 2 to consider both my_agent's and opponent's actions
        self.group_memory_length = self.current_memory_max_length
        # list of groups of memory patterns
        self.groups_of_memory_patterns = []
        for i in range(STEPS_MAX, STEPS_MIN - 1, -1):
            self.groups_of_memory_patterns.append(
                {
                    # how many steps in a row are in the pattern
                    "memory_length": self.group_memory_length,
                    # list of memory patterns
                    "memory_patterns": [],
                }
            )
            self.group_memory_length -= 2

    def _get_probs(self, step: int, score: int, history: pd.DataFrame) -> np.ndarray:
        last_action = int(history.loc[step - 1, "action"]) if len(history) > 0 else -1
        # add my_agent's current step to current_memory
        if len(history) > 0:
            last_action = self.previous_action["action"]
            self.update_current_memory(last_action)
            if self.previous_action["action"] != last_action:
                self.previous_action["action"] = last_action
                self.previous_action["action_from_pattern"] = False
                self.previous_action["pattern_group_index"] = None
                self.previous_action["pattern_index"] = None
        """ your ad here """
        # action of my_agent
        my_action = None
        # Removed the random action
        # if it's not first step
        if len(history) > 0:
            # add opponent's last step to current_memory
            self.current_memory.append(int(history.loc[step - 1, "opponent_action"]))
            # previous step won or lost
            previous_step_result = self.get_step_result_for_my_agent(
                last_action, int(history.loc[step - 1, "opponent_action"])
            )
            self.reward += previous_step_result
            # if previous action of my_agent was taken from pattern
            if self.previous_action["action_from_pattern"]:
                self.evaluate_pattern_efficiency(previous_step_result)

        for i in range(len(self.groups_of_memory_patterns)):
            # if possible, update or add some memory pattern in this group
            self.update_memory_pattern(self.groups_of_memory_patterns[i], history, step)
            # if action was not yet found
            if my_action is None:
                my_action, pattern_index = self.find_action(
                    self.groups_of_memory_patterns[i], i
                )
                if my_action is not None:
                    # save action's data
                    self.previous_action["action"] = my_action
                    self.previous_action["action_from_pattern"] = True
                    self.previous_action["pattern_group_index"] = i
                    self.previous_action["pattern_index"] = pattern_index

        # if no action was found
        if my_action is None:
            # choose action randomly
            my_action = random.randint(0, 2)
            # save action's data
            self.previous_action["action"] = my_action
            self.previous_action["action_from_pattern"] = False
            self.previous_action["pattern_group_index"] = None
            self.previous_action["pattern_index"] = None
        return one_hot(my_action)

    def evaluate_pattern_efficiency(self, previous_step_result):
        """
        evaluate efficiency of the pattern and, if pattern is inefficient,
        remove it from agent's memory
        """
        pattern_group_index = self.previous_action["pattern_group_index"]
        pattern_index = self.previous_action["pattern_index"]
        pattern = self.groups_of_memory_patterns[pattern_group_index][
            "memory_patterns"
        ][pattern_index]
        pattern["reward"] += previous_step_result
        # if pattern is inefficient
        if pattern["reward"] <= EFFICIENCY_THRESHOLD:
            # remove pattern from agent's memory
            del self.groups_of_memory_patterns[pattern_group_index]["memory_patterns"][
                pattern_index
            ]

    def find_action(self, group, group_index):
        """ if possible, find my_action in this group of memory patterns """
        if len(self.current_memory) > group["memory_length"]:
            this_step_memory = self.current_memory[-group["memory_length"] :]
            memory_pattern, pattern_index = self.find_pattern(
                group["memory_patterns"], this_step_memory, group["memory_length"]
            )
            if memory_pattern is not None:
                my_action_amount = 0
                for action in memory_pattern["opp_next_actions"]:
                    # if this opponent's action occurred more times than currently chosen action
                    # or, if it occured the same amount of times and this one is choosen randomly among them
                    if action["amount"] > my_action_amount or (
                        action["amount"] == my_action_amount and random.random() > 0.5
                    ):
                        my_action_amount = action["amount"]
                        my_action = action["response"]
                return my_action, pattern_index
        return None, None

    def find_pattern(self, memory_patterns, memory, memory_length):
        """ find appropriate pattern and its index in memory """
        for i in range(len(memory_patterns)):
            actions_matched = 0
            for j in range(memory_length):
                if memory_patterns[i]["actions"][j] == memory[j]:
                    actions_matched += 1
                else:
                    break
            # if memory fits this pattern
            if actions_matched == memory_length:
                return memory_patterns[i], i
        # appropriate pattern not found
        return None, None

    def get_step_result_for_my_agent(self, my_agent_action, opp_action):
        """
        get result of the step for my_agent
        1, 0 and -1 representing win, tie and lost results of the game respectively
        reward will be taken from observation in the next release of kaggle environments
        """
        if my_agent_action == opp_action:
            return 0
        elif (my_agent_action == (opp_action + 1)) or (
            my_agent_action == 0 and opp_action == 2
        ):
            return 1
        else:
            return -1

    def update_current_memory(self, my_action):
        """ add my_agent's current step to current_memory """
        # if there's too many actions in the current_memory
        if len(self.current_memory) > self.current_memory_max_length:
            # delete first two elements in current memory
            # (actions of the oldest step in current memory)
            del self.current_memory[:2]
        # add agent's last action to agent's current memory
        self.current_memory.append(my_action)

    def update_memory_pattern(self, group, history, step):
        """ if possible, update or add some memory pattern in this group """
        # if length of current memory is suitable for this group of memory patterns
        if len(self.current_memory) > group["memory_length"]:
            # get memory of the previous step
            # considering that last step actions of both agents are already present in current_memory
            previous_step_memory = self.current_memory[-group["memory_length"] - 2 : -2]
            previous_pattern, pattern_index = self.find_pattern(
                group["memory_patterns"], previous_step_memory, group["memory_length"]
            )
            if previous_pattern is None:
                previous_pattern = {
                    # list of actions of both players
                    "actions": previous_step_memory.copy(),
                    # total reward earned by using this pattern
                    "reward": 0,
                    # list of observed opponent's actions after each occurrence of this pattern
                    "opp_next_actions": [
                        # action that was made by opponent,
                        # amount of times that action occurred,
                        # what should be the response of my_agent
                        {"action": 0, "amount": 0, "response": 1},
                        {"action": 1, "amount": 0, "response": 2},
                        {"action": 2, "amount": 0, "response": 0},
                    ],
                }
                group["memory_patterns"].append(previous_pattern)
            # update previous_pattern
            for action in previous_pattern["opp_next_actions"]:
                if action["action"] == int(history.loc[step - 1, "opponent_action"]):
                    action["amount"] += 1
