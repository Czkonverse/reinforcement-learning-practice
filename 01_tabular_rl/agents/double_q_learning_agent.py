import numpy as np


class DoubleQLearning:
    def __init__(self, alpha=0.1, gamma=1.0, epsilon=0.1, seed=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table_1 = {0: np.zeros(2), 1: np.zeros(10)}
        self.q_table_2 = {0: np.zeros(2), 1: np.zeros(10)}

        self.rng = np.random.default_rng(seed)

    def select_action(self, state, training=False):
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(self.q_table_1[state].shape[0])

        q_1_values = self.q_table_1[state]
        q_2_values = self.q_table_2[state]
        q_combine = q_1_values + q_2_values
        best_actions = np.flatnonzero(q_combine == np.max(q_combine))
        return self.rng.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        update_num = self.rng.integers(2)

        if update_num == 0:
            old_q = self.q_table_1[state][action]
            if done:
                target = reward
            else:
                # Q1 execute
                q_1_values = self.q_table_1[next_state]
                best_actions = np.flatnonzero(q_1_values == np.max(q_1_values))
                best_action = self.rng.choice(best_actions)
                # Q2 evaluates the action selected by Q1
                target = reward + self.gamma * self.q_table_2[next_state][best_action]

            self.q_table_1[state][action] = old_q + self.alpha * (target - old_q)
        else:
            old_q = self.q_table_2[state][action]
            if done:
                target = reward
            else:
                # Q2 chooses the best action
                q_2_values = self.q_table_2[next_state]
                best_actions = np.flatnonzero(q_2_values == np.max(q_2_values))
                best_action = self.rng.choice(best_actions)

                # Q1 evaluates the action selected by Q2
                target = reward + self.gamma * self.q_table_1[next_state][best_action]

            self.q_table_2[state][action] = old_q + self.alpha * (target - old_q)
