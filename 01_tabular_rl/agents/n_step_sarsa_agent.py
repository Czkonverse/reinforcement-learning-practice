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

        self._states = [state]
        self._actions = [self.select_action(state)]
        self._rewards = []
        return self._actions[0]

    def step(self, reward, next_state, terminal, truncated):

        self._rewards.append(reward)
        self._states.append(next_state)

        if terminal:
            # No action follows a terminal state; the tail is flushed by
            # end_episode("terminal").
            return None

        next_action = self.select_action(next_state)
        self._actions.append(next_action)

        if truncated:
            # Max-step limit hit: this transition and the rest of the tail are
            # flushed with bootstrapping by end_episode("truncated").
            return next_action

        # normal case: update transition tau = t - n + 1 if it has n rewards
        t = len(self._rewards) - 1
        tau = t - self.n + 1

        if tau >= 0:
            G = self._discounted_rewards(tau, tau + self.n)

            # bootstrap: + gamma^n * Q(S_{tau+n}, A_{tau+n})
            bootstrap_state = self._states[tau + self.n]
            bootstrap_action = self._actions[tau + self.n]
            G += self.gamma**self.n * self.q_table[bootstrap_state][bootstrap_action]

            self._update_Q(tau, G)

        return next_action

    def select_action(self, state, training=True):
        # epsilon-greedy
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))

        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return int(self.rng.choice(best_actions))

    def end_episode(self, reason):
        """Finish the current episode by flushing the remaining tail transitions.

        reason == "terminal"  : the episode truly ended (e.g. goal reached);
                                the tail uses the pure discounted return.
        reason == "truncated" : the max-step limit was hit, which is NOT a real
                                terminal, so the tail bootstraps to Q(S_T, A_T).
        """
        if reason == "terminal":
            self._flush_terminal()
        elif reason == "truncated":
            self._flush_truncated()
        else:
            raise ValueError(f"Illegal reason: {reason}")

        # clear the buffers, ready for the next episode
        self._states = []
        self._actions = []
        self._rewards = []

    def _discounted_rewards(self, start, end):
        """Sum of gamma^(i-start) * rewards[i] for i in [start, end).

        rewards[i] is the reward for transition i (i.e. R_{i+1} of the book).
        """
        G = 0.0
        for i in range(start, end):
            G += self.gamma ** (i - start) * self._rewards[i]
        return G

    def _update_Q(self, tau, target):
        """Q(S_tau, A_tau) <- Q(S_tau, A_tau) + alpha * (target - Q(S_tau, A_tau))."""
        state = self._states[tau]
        action = self._actions[tau]
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.alpha * (target - old_q)

    def _flush_terminal(self):
        """Terminal tail: pure discounted return, no bootstrap (V(terminal)=0).

        Example (n=3):
            Q(S0,A0) <- R1 + gamma R2 + gamma^2 R3   (episode ends after R3)
        """
        T = len(self._rewards)
        first_tau = max(0, T - self.n)
        for tau in range(first_tau, T):
            G = self._discounted_rewards(tau, T)
            self._update_Q(tau, G)

    def _flush_truncated(self):
        """Truncated tail: like the terminal case, but the last state is not
        absorbing, so we bootstrap the tail to Q(S_T, A_T):

            G_tau = R_{tau+1} + ... + gamma^{T-tau-1} R_T
                    + gamma^{T-tau} Q(S_T, A_T)

        Example (n=3, T=3):
            Q(S0,A0) <- R1 + gamma R2 + gamma^2 R3 + gamma^3 Q(S3, A3)
        """
        T = len(self._rewards)
        first_tau = max(0, T - self.n)
        final_state = self._states[T]
        final_action = self._actions[T]
        for tau in range(first_tau, T):
            G = self._discounted_rewards(tau, T)
            G += self.gamma ** (T - tau) * self.q_table[final_state][final_action]
            self._update_Q(tau, G)
