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
        #

        self._states = [state]
        self._actions = [self.select_action(state)]
        self._rewards = []
        return self._actions[0]

    def step(self, reward, next_state, terminal, truncated):

        self._rewards.append(reward)
        self._states.append(next_state)

        if terminal:
            return None

        next_action = self.select_action(next_state)

        self._actions.append(next_action)

        if truncated:
            return next_action

        # normal case
        t = len(self._rewards) - 1

        tau = t - self.n + 1

        if tau >= 0:
            G = 0.0

            for i in range(tau, tau + self.n):
                G += self.gamma ** (i - tau) * self._rewards[i]

        # bootstrap : + gamma^n Q(S_{tau+n}, A_{tau+n})
        bootstrap_state = self._states[tau + self.n]
        bootstrap_action = self._actions[tau + self.n]

        G += self.gamma**self.n * self.q_table[bootstrap_state][bootstrap_action]

        # update Q(S_tau, A_tau)
        state = self._states[tau]
        action = self._actions[tau]

        old_q = self.q_table[state][action]

        self.q_table[state][action] = old_q + self.alpha * (G - old_q)

        return next_action

    def select_action(self, state, training=True):
        # epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return np.random.choice(best_actions)

    def end_episode(self, state):
        T = len(self._rewards)

        if state == "terminal":
            pass
        elif state == "truncated":
            pass
        else:
            raise ValueError(f"Illegal state: : {state}")

    def _flush_truncated(self):
        # R0 = R1 + gamma ** 1 * R2 + gamma ** 2 * R3 + gamma ** 3 * Q(S3, A3)

        T = len(self._rewards)

        first_tau = max(0, T - self.n + 1)

        for flush_tau in range(first_tau, T):

            G = 0.0

            # reward
            for i in range(flush_tau, T):
                G += self.gamma ** (i - flush_tau) * self._rewards[i]

            # bootstrap - Q(S_T, A_T)
            final_state = self._states[T]
            final_action = self._actions[T]

            G += self.gamma ** (T - flush_tau) * self.q_table[final_state][final_action]

            # update Q(S_tau, A_tau)
            state = self._states[flush_tau]
            action = self._actions[flush_tau]

            old_q = self.q_table[state][action]

            self.q_table[state][action] = old_q + self.alpha * (G - old_q)

    def _flush_terminal(self):

        T = len(self._rewards)

        first_tau = max(0, T - self.n + 1)

        for flush_tau in range(first_tau, T):
            G = 0.0

            # reward
            for i in range(flush_tau, T):
                G += self.gamma ** (i - flush_tau) * self._rewards[i]

            # update Q(S_tau, A_tau)
            state = self._states[flush_tau]
            action = self._actions[flush_tau]

            old_q = self.q_table[state][action]

            self.q_table[state][action] = old_q + self.alpha * (G - old_q)
