import numpy as np


class NStepSarsaAgent:
    """n 步 SARSA 智能体（直观版）。

    相比 v2：不再把逻辑拆成 begin_episode / learn / end_episode /
    _discounted_return / _update_at 一堆小函数，而是把整个流程摊平，
    更新公式直接逐行写在代码里：

        Q(S_t, A_t) <- Q(S_t, A_t) + alpha * (G_{t:t+n} - Q(S_t, A_t))

        G_{t:t+n} = R_{t+1} + gamma*R_{t+2} + ... + gamma^{n-1}*R_{t+n}
                    + gamma^n * Q(S_{t+n}, A_{t+n})

    时序约定（下标从 0 开始）：
      start_episode(S0)         记录 S0，返回初始动作 A0；
      step(R_k, S_k, done)      表示从 S_{k-1} 走 A_{k-1} 到达 S_k 并得到 R_k，
                                此时 rewards 下标 k-1 存 R_k，states 下标 k 存 S_k；
      每个 episode 结束时缓冲区自动清空；若因步数上限被截断（done 一直为 False），
      训练代码需额外调用一次 finish_episode() 补完尾部。
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

        # 当前 episode 的轨迹缓冲
        self.states = []
        self.actions = []
        self.rewards = []

    def select_action(self, state, training=True):
        # epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return np.random.choice(best_actions)

    def start_episode(self, state):
        """开始新 episode：记录 S0，返回初始动作 A0。"""
        self.states = [state]
        self.actions = [self.select_action(state)]
        self.rewards = []
        return self.actions[0]

    def step(self, reward, next_state, done):
        """每走一步调用一次，返回下一步动作（终止时返回 None）。

        依次做三件事：
          1. 把这一步的 (R_t, S_t) 记入缓冲；未终止时再选并记下 A_t；
          2. 若 tau = t - n + 1 >= 0 且未终止，用 n 步回报
             （bootstrap 到 Q(S_{t+n}, A_{t+n})）更新这一跳；
          3. 若 done，把剩下的尾部 transition 用「到终点为止」的纯回报补完，
             然后清空缓冲。
        """
        # ---- 1. 记录这一步 ----
        self.rewards.append(reward)
        self.states.append(next_state)
        if done:
            next_action = None
        else:
            next_action = self.select_action(next_state)
            self.actions.append(next_action)

        # ---- 2. n 步更新：更新「n 步之前」那一跳 ----
        t = len(self.rewards) - 1
        tau = t - self.n + 1
        if not done and tau >= 0:
            # G = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n}
            G = 0.0
            for i in range(tau, tau + self.n):
                G += self.gamma ** (i - tau) * self.rewards[i]
            # 之后再用 γ^n * Q(S_{t+n}, A_{t+n}) 引导
            G += (
                self.gamma**self.n
                * self.q_table[self.states[tau + self.n]][self.actions[tau + self.n]]
            )

            s, a = self.states[tau], self.actions[tau]
            self.q_table[s][a] += self.alpha * (G - self.q_table[s][a])

        # ---- 3. 到达终点：补完剩余跳数并清空 ----
        if done:
            T = len(self.rewards)
            for flush_tau in range(max(0, T - self.n), T):
                # 终点之后没有下一步了，直接用纯折扣回报
                G = 0.0
                for i in range(flush_tau, T):
                    G += self.gamma ** (i - flush_tau) * self.rewards[i]

                s, a = self.states[flush_tau], self.actions[flush_tau]
                self.q_table[s][a] += self.alpha * (G - self.q_table[s][a])

            self.states, self.actions, self.rewards = [], [], []

        return next_action

    def finish_episode(self):
        """（截断时调用）手动结束 episode：补完缓冲里剩余的尾部 transition。

        截断时最后一次 step 的 done=False，已经用 bootstrap 更新过
        tau = T - n 那一跳，所以这里从 T - n + 1 开始补，避免重复更新；
        正常终止时 step 内部已自动补完并清空，缓冲区为空，本方法直接返回。
        """
        T = len(self.rewards)
        for flush_tau in range(max(0, T - self.n + 1), T):
            G = 0.0
            for i in range(flush_tau, T):
                G += self.gamma ** (i - flush_tau) * self.rewards[i]

            s, a = self.states[flush_tau], self.actions[flush_tau]
            self.q_table[s][a] += self.alpha * (G - self.q_table[s][a])

        self.states, self.actions, self.rewards = [], [], []
