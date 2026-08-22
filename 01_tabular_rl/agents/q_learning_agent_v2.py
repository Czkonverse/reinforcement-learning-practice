import numpy as np


class QLearningAgentV2:
    def __init__(self, alpha=0.1, gamma=1.0, epsilon=0.2, seed=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = {0: np.zeros(2), 1: np.zeros(10)}
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, training=True):
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(self.q_table[state].shape[0])
        else:
            q_values = self.q_table[state]
            best_actions = np.flatnonzero(q_values == np.max(q_values))
            return self.rng.choice(best_actions)

    def update(self, state, action, next_state, reward, done):

        old_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] = old_q + self.alpha * (target - old_q)
