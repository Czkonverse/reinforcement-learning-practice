import numpy as np


class NStepSarsaAgent:
    """n 步 SARSA 智能体（重构版）。

    相比 v1（n_step_sarsa_agent.py）的改进：
    - n-step 的 episode 缓冲（states / actions / rewards）、
      回报计算与终止时的 flush 逻辑全部封装在智能体内部；
    - experiment 只需按  begin_episode -> learn -> (end_episode) 驱动，
      不需要自己维护索引，避免常见的下标越界 / 变量遮蔽类 bug。

    时序约定（以 0 为起点）：
      begin_episode(S0) 记录并返回 A0；
      第 k 次 learn(R_k, S_k, done) 处理 transition (S_{k-1}, A_{k-1}) -> (R_k, S_k)，
      其中 R_k 记为 rewards[k-1]，S_k 记为 states[k]，A_{k-1} 记为 actions[k-1]。
    """

    def __init__(
        self, state_shape, num_actions, n=3, alpha=0.1, gamma=0.9, epsilon=0.2
    ):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.q_table = np.zeros((*state_shape, num_actions))

        # episode 缓冲
        self._states = []
        self._actions = []
        self._rewards = []

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def begin_episode(self, state):
        """开始一个新 episode：清空缓冲，记录 S0，选择并返回初始动作 A0。"""
        self._states = [state]
        self._actions = [self.select_action(state)]
        self._rewards = []
        return self._actions[0]

    def learn(self, reward, next_state, done):
        """在 env.step(action) 之后调用一次，处理一个 transition。

        内部完成三件事：
        1) 缓存 reward / next_state；若未终止，选并缓存 next_action；
        2) 若 tau = t - n + 1 >= 0 且未终止，执行一次 n-step 更新；
        3) 若 done，对剩余未更新项 flush 到 episode 终点并清空缓冲。

        返回：下一步应执行的动作；episode 已终止时返回 None。
        """
        self._rewards.append(reward)
        self._states.append(next_state)

        if not done:
            next_action = self.select_action(next_state)
            self._actions.append(next_action)
        else:
            next_action = None

        t = len(self._rewards) - 1  # 本次 transition 的下标（0 起）
        tau = t - self.n + 1  # 本次可以更新的 transition 下标

        if not done and tau >= 0:
            self._update_at(tau, bootstrap=True)

        if done:
            self.end_episode()

        return next_action

    def end_episode(self, truncated=False):
        """episode 结束时调用：flush 剩余未更新项。

        区分两种结束方式（决定 flush 起点，避免同一 transition 被更新两次）：
        - truncated=False（正常终止，learn 内部 done=True 时自动调用）：
          最后一跳 learn 未做 bootstrap 更新，需 flush tau in [max(0,T-n), T-1]。
        - truncated=True（达到步数上限被截断，最后一跳 learn 的 done=False，
          由实验代码在循环结束后调用）：
          最后一次 learn 已用 bootstrap 更新过 tau=T-n（bootstrap 到 S_T, A_T），
          这里只需 flush tau in [max(0,T-n+1), T-1]，跳过 T-n 避免重复更新。
        之后清空缓冲等待下一个 episode。
        """
        T = len(self._rewards)
        if truncated:
            start = max(0, T - self.n + 1)
        else:
            start = max(0, T - self.n)
        for flush_tau in range(start, T):
            self._update_at(flush_tau, bootstrap=False)

        self._states = []
        self._actions = []
        self._rewards = []

    def select_action(self, state, training=True):
        # epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return np.random.choice(best_actions)

    def update(self, state, action, target):
        """Q 值更新：Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))"""
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.alpha * (target - old_q)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _discounted_return(self, start, end):
        """从 rewards[start] 累加到 rewards[end-1] 的折扣回报。"""
        G = 0.0
        for i in range(start, end):
            G += self.gamma ** (i - start) * self._rewards[i]
        return G

    def _update_at(self, tau, bootstrap):
        """更新 (S_tau, A_tau)。

        bootstrap=True : G = R_{tau+1} + ... + gamma^(n-1) R_{tau+n}
                         + gamma^n * Q(S_{tau+n}, A_{tau+n})
        bootstrap=False: G = R_{tau+1} + ... + gamma^(T-tau-1) R_T
                         （episode 已终止 / 截断，无需 bootstrap）
        """
        if bootstrap:
            end = tau + self.n
            G = self._discounted_return(tau, end)
            G += (
                self.gamma**self.n * self.q_table[self._states[end]][self._actions[end]]
            )
        else:
            end = len(self._rewards)
            G = self._discounted_return(tau, end)

        self.update(self._states[tau], self._actions[tau], G)
