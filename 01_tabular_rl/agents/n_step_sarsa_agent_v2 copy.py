import numpy as np

# S0 --A0--> S1 --A1--> S2 --A2--> S3--A3--> Terminal
#      R1         R2         R3        R4

# update, Q(S0​,A0​) --> R1​+γR2​+(γ)^2 * R3​+(γ)^3 * Q(S3​,A3​)


class NStepSarsaAgent:

    def __init__(
        self, state_shape, num_actions, n=3, alpha=0.1, gamma=0.9, epsilon=0.2
    ):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.q_table = np.zeros_like((*state_shape, num_actions))

        # buffer for necessary results
        self._states = []
        self._actions = []
        self._rewards = []

    def select_action(self, state, training=True):

        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return np.random.choice(best_actions)

    def update(self, state, action, target):
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.alpha * (target - old_q)

    def begin_episode(self, state):
        self._states = [state]
        self._actions = [self.select_action(state)]
        self._rewards = []

        return self._actions[0]

    def learn(self, reward, next_state, done):
        self._rewards.append(reward)
        self._states.append(next_state)

        if not done:
            next_action = self.select_action(next_state)
            self._actions.append(next_action)
        else:
            next_action = None

        t = len(self._rewards) - 1  # index of this transition
        tau = t - self.n + 1
        if not done and tau >= 0:
            self._update_at(tau, bootstrap=True)

        if done:
            self.end_episode()

        return next_action

    def end_episode(self):
        # case 1: within max_steps limitation, episode ends normally
        # case 2: out of max_steps, end episode directly
        # both cases need flush, update the q_table for undoing states and actions
        # and reset the buffer of this turn episode
        T = len(self._rewards)
        start_tau = max(0, T - self.n)
        for flush_tau in range(start_tau, T):
            self._update_at(flush_tau, bootstrap=False)

        self._states = []
        self._actions = []
        self._rewards = []

    def _discounted_return(self, start, end):
        # γ^0 * R1​ + γ^1 * R2​ + γ^2 * R3
        G = 0.0
        for i in range(start, end):
            G += self.gamma ** (i - start) * self._rewards[i]

        return G

    def _update_at(self, tau, bootstrap):
        if bootstrap:
            # γ^0 * R1​ + γ^1 * R2​ + γ^2 * R3
            end = tau + self.n
            G = self._discounted_return(start=tau, end=end)
            # γ^3 * Q(S3​,A3​)
            G += (
                self.gamma**self.n * self.q_table[self._states[end]][self._actions[end]]
            )
        else:
            end = len(self._rewards)
            G = self._discounted_return(start=tau, end=end)

        self.update(self._states[tau], self._actions[tau], G)
