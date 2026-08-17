import numpy as np


class RandomAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def select_action(self, state=None, training=True):
        return np.random.randint(self.num_actions)
