import numpy as np


class NStepSarsaAgent:
    def __init__(
        self, state_shape, num_actions, n=3, alpha=0.1, gamma=0.9, epsilon=0.2
    ):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.q_table = np.zeros((*state_shape, num_actions))

    def select_action(self, state, training=True):
        # epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return np.random.choice(best_actions)

    def update(self, state, action, target):
        old_q = self.q_table[state][action]

        self.q_table[state][action] = old_q + self.alpha * (target - old_q)
