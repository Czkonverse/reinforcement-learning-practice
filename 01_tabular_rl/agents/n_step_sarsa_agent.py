import numpy as np


# Answer these questions below:
# 1 What are T, t, and tau?
# 2 Why is tau = t - n + 1?
# 3 If an episode terminates, how do we update the Q table?
# 4 If an episode is truncated at the fixed max step limit, how do we update the Q table? What is the difference from Question 3?
# 5 How many update equations exist? What are the update equations in different cases?
class NStepSarsaAgent:
    def __init__(
        self,
        state_shape,
        num_actions,
        n=3,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2,
        seed=None,
    ):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.q_table = np.zeros((*state_shape, num_actions))

        self.rng = np.random.default_rng(seed)

        # buffers of states, actions, rewards
        self._states = []
        self._actions = []
        self._rewards = []

    def begin_episode(self, state):
        pass

    def learn(
        self,
        reward,
        next_state,
    ):
        pass

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
