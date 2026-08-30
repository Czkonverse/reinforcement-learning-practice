import numpy as np


class DynaQAgent:
    """Dyna-Q 智能体（直观版）。

    在 Q-learning 的基础上加了两样东西：
      1. 模型（日记本）：把每一条真实经验 (S,A) -> (R,S',done) 记下来，
         Model[(S,A)] = (R, S', done)；
      2. 规划（脑内练习）：每走完一个真实步骤，从日记本里随机抽 n 条
         旧经验，用和 Q-learning 完全相同的公式各更新一次 Q。

    一句话：真实经验只有一份，但通过规划，它的价值被放大了 n 倍。

    更新公式（真实学习与脑内规划共用同一条）：

        Q(S,A) <- Q(S,A) + alpha * ( R + gamma * max_a' Q(S',a') - Q(S,A) )
        若该步 done，则不再 bootstrap：target = R。

    时序约定（与 n_step_sarsa_agent_v3 一致）：
      start_episode(S0)      记住 S0，返回初始动作 A0；
      step(R_k, S_k, done)   表示从 S_{k-1} 走 A_{k-1} 到达 S_k 并得到 R_k，
                             内部完成 真实学习 -> 记日记 -> n 次脑内规划，
                             并返回下一步动作（终止时返回 None）。

    因为是 1 步算法，不需要 episode 缓冲，也没有尾部 transition 要补；
    finish_episode 只是为了让训练循环与 n-step 版保持同构而保留的空方法。
    """

    def __init__(
        self, state_shape, num_actions, n=50, alpha=0.1, gamma=0.9, epsilon=0.2
    ):
        self.n = n  # 每个真实步骤后做的规划（模拟更新）次数
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.q_table = np.zeros((*state_shape, num_actions))

        # 模型（日记本）：Model[(s, a)] = (reward, next_state, done)
        self.model = {}
        # 所有见过的 (s, a)，供规划时随机采样（保证一定能从模型里查到）
        self.seen_transitions = []

        # 当前状态 / 上一个动作（1 步更新需要记忆）
        self.last_state = None
        self.last_action = None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def select_action(self, state, training=True):
        # epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return np.random.choice(best_actions)

    def start_episode(self, state):
        """开始新 episode：记住 S0，返回初始动作 A0。"""
        self.last_state = state
        self.last_action = self.select_action(state)
        return self.last_action

    def step(self, reward, next_state, done):
        """每走一个真实步骤调用一次，返回下一步动作（终止时返回 None）。

        依次做三件事：
          1. 真实学习：用 Q-learning 公式更新刚走的这一跳 Q(S,A)；
          2. 记日记：  把 (S,A) -> (R, S', done) 写进模型；
          3. 脑内规划：从日记本随机抽 n 条旧经验，各做一次 Q-learning 更新。
        """
        s, a = self.last_state, self.last_action

        # ---- 1. 真实学习 ----
        self._q_update(s, a, reward, next_state, done)

        # ---- 2. 记日记 ----
        self._remember(s, a, reward, next_state, done)

        # ---- 3. 脑内规划 n 次 ----
        for _ in range(self.n):
            self._planning_step()

        # ---- 选下一步动作 ----
        if done:
            self.last_action = None
            return None

        self.last_state = next_state
        self.last_action = self.select_action(next_state)
        return self.last_action

    def finish_episode(self):
        """空方法：Dyna-Q 是 1 步算法，每个真实步骤都已即时更新，
        没有像 n-step 那样的尾部 transition 需要补，故无需任何操作。
        保留它只是为了与 n_step_sarsa_agent_v3 的训练循环保持同构。"""
        pass

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _q_update(self, s, a, reward, next_state, done):
        """Q-learning 更新。

        done=False: Q(s,a) <- Q(s,a) + alpha*(R + gamma*max_a' Q(s',a') - Q(s,a))
        done=True : Q(s,a) <- Q(s,a) + alpha*(R - Q(s,a))   （终点不再 bootstrap）
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[s][a] += self.alpha * (target - self.q_table[s][a])

    def _remember(self, s, a, reward, next_state, done):
        """记日记：写入 Model[(s,a)] = (R, s', done)，并登记这条 (s,a) 供规划采样。"""
        key = (s, a)
        if key not in self.model:
            self.seen_transitions.append(key)
        self.model[key] = (reward, next_state, done)

    def _planning_step(self):
        """脑内规划一次：随机抽一条见过的 (s,a)，从模型读出 (R,s',done)，
        做一次与真实学习完全相同的 Q-learning 更新。"""
        if not self.seen_transitions:
            return

        idx = np.random.randint(len(self.seen_transitions))
        s, a = self.seen_transitions[idx]
        reward, next_state, done = self.model[(s, a)]

        self._q_update(s, a, reward, next_state, done)
