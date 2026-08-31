import numpy as np


class DynaQAgentTemplate:
    """Dyna-Q 智能体（补全练习版）。

    请先阅读 docs/dyna_q_tutorial.md，再动手补全本文件中的 TODO。
    补全顺序建议：1 -> 4 -> 5 -> 6 -> 2 -> 3。
    补全后运行本文件可做自检，也可对照参考答案 agents/dyna_q_agent.py。

    核心思想（一句话）：Q-learning + 日记本(model) + 每步脑内重放 n 次。

    更新公式（真实学习与脑内规划共用同一条）：
        Q(s,a) <- Q(s,a) + alpha * (R + gamma * max_a' Q(s',a') - Q(s,a))
        若 done 为 True：target = R（不再 bootstrap）。
    """

    def __init__(
        self, state_shape, num_actions, n=50, alpha=0.1, gamma=0.9, epsilon=0.2
    ):
        """TODO 1: 初始化所有状态。

        需要准备：
          1. 把 n / alpha / gamma / epsilon / num_actions 存到 self 上；
          2. self.q_table：形状 (*state_shape, num_actions) 的全 0 数组
             （提示：np.zeros((*state_shape, num_actions))）；
          3. self.model：空字典，之后存 Model[(s,a)] = (reward, next_state, done)；
          4. self.seen_transitions：空列表，记录所有见过的 (s,a) 对，
             供规划时随机采样（保证能查到模型）；
          5. self.last_state / self.last_action：先置 None，
             之后 start_episode / step 会用到（1 步更新需要记住刚走过的 (s,a)）。
        """
        raise NotImplementedError("TODO 1: 请补全 __init__")

    def select_action(self, state, training=True):
        """epsilon-greedy 选动作（已给出，和之前所有 agent 一样，无需修改）。"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return np.random.choice(best_actions)

    def start_episode(self, state):
        """TODO 2: 开始一个新 episode。

        提示：记住当前状态 self.last_state = state，
        再用 select_action 选初始动作存到 self.last_action，并把它返回给训练循环。
        """
        raise NotImplementedError("TODO 2: 请补全 start_episode")

    def step(self, reward, next_state, done):
        """TODO 3: 每走一个真实步骤调用一次，返回下一步动作（done 时返回 None）。

        流程（对着伪代码做）：
          1) 真实学习：用刚走的这一跳 (self.last_state, self.last_action,
             reward, next_state, done) 调 self._q_update(...)；
          2) 记日记：  把这一跳调 self._remember(...) 写进模型；
          3) 脑内规划：for _ in range(self.n): self._planning_step()；
          4) 选下一步：若 done 返回 None；否则更新 self.last_state，
             用 select_action 选动作存到 self.last_action 并返回它。

        提示：先取出 s, a = self.last_state, self.last_action。
        """
        raise NotImplementedError("TODO 3: 请补全 step")

    def finish_episode(self):
        """空方法：1 步算法没有尾部 transition 要补。
        保留它只是为了和 n-step 版的训练循环保持同构。"""
        pass

    def _q_update(self, s, a, reward, next_state, done):
        """TODO 4: Q-learning 更新（真实学习与脑内规划共用）。

        公式：
            target = reward                                   (done=True)
            target = reward + gamma * max(Q(s', :))           (done=False)
            Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))

        提示：max 可以用 np.max(self.q_table[next_state])。
        """
        raise NotImplementedError("TODO 4: 请补全 _q_update")

    def _remember(self, s, a, reward, next_state, done):
        """TODO 5: 记日记 —— 把一条真实经验写进模型。

        提示：
          key = (s, a)
          1. 若 key 还没出现过（not in self.model），先把它加进
             self.seen_transitions（保证规划能随机抽到它）；
          2. 然后写模型：self.model[key] = (reward, next_state, done)。
        """
        raise NotImplementedError("TODO 5: 请补全 _remember")

    def _planning_step(self):
        """TODO 6: 脑内规划一次 —— 用模型"重演"一条旧经验并更新 Q。

        提示：
          1. 若 self.seen_transitions 为空，直接 return（还没有经验可规划）；
          2. 随机抽一条：idx = np.random.randint(len(self.seen_transitions))，
             s, a = self.seen_transitions[idx]；
          3. 从模型读出 reward, next_state, done = self.model[(s, a)]；
          4. 用和真实学习完全相同的 self._q_update(s, a, reward, next_state, done)。
        """
        raise NotImplementedError("TODO 6: 请补全 _planning_step")


if __name__ == "__main__":
    # ===== 补全后的自检：在 CliffWorld 上跑 100 局 =====
    # 填完所有 TODO 后再运行本文件：
    #   python agents/dyna_q_agent_template.py
    # 期望看到：avg reward 明显优于随机基线（约 -100 左右），
    # 且比 n=0（即纯 Q-learning）学得快。
    import os
    import sys

    # 保证无论从哪个目录运行都能找到 envs 包
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from envs.cliff_world import CliffWorld

    env = CliffWorld()
    agent = DynaQAgentTemplate(
        (env.rows, env.cols), env.num_actions, n=50
    )

    num_episodes = 100
    max_step = 200
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        action = agent.start_episode(state)
        total = 0.0
        done = False
        for _ in range(max_step):
            next_state, reward, done = env.step(action)
            total += reward
            action = agent.step(reward, next_state, done)
            if done:
                break
        rewards.append(total)

    print("avg reward over %d episodes: %.2f" % (num_episodes, np.mean(rewards)))
