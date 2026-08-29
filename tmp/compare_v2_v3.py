"""临时对比脚本：验证 v3 与 v2 的 n-step SARSA 行为一致。"""
import sys

import numpy as np

sys.path.insert(0, "01_tabular_rl")

from agents.n_step_sarsa_agent_v2 import NStepSarsaAgent as AgentV2
from agents.n_step_sarsa_agent_v3 import NStepSarsaAgent as AgentV3
from envs.cliff_world import CliffWorld


def run_with(agent_cls, seed, num_episodes=200, max_step=200):
    np.random.seed(seed)
    env = CliffWorld()
    agent = agent_cls(
        (env.rows, env.cols),
        env.num_actions,
        n=3,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2,
    )

    rewards_all = []
    steps_all = []
    success_all = []
    for _ in range(num_episodes):
        state = env.reset()
        action = agent.start_episode(state) if hasattr(agent, "start_episode") else agent.begin_episode(state)

        total_reward = 0.0
        done = False
        for step in range(max_step):
            next_state, reward, done = env.step(action)
            total_reward += reward

            if hasattr(agent, "step"):
                action = agent.step(reward, next_state, done)
            else:
                action = agent.learn(reward, next_state, done)

            if done:
                break

        if not done:
            if hasattr(agent, "finish_episode"):
                agent.finish_episode()
            else:
                agent.end_episode(truncated=True)

        rewards_all.append(total_reward)
        steps_all.append(step + 1)
        success_all.append(done)

    return agent.q_table, np.array(rewards_all), np.array(steps_all), np.array(success_all)


if __name__ == "__main__":
    for seed in [0, 1, 42]:
        q2, r2, s2, succ2 = run_with(AgentV2, seed)
        q3, r3, s3, succ3 = run_with(AgentV3, seed)

        q_diff = np.abs(q2 - q3).max()
        r_diff = np.abs(r2 - r3).max()
        s_diff = np.abs(s2 - s3).max()
        succ_same = np.array_equal(succ2, succ3)

        print(f"seed={seed}: q_diff={q_diff:.2e}  r_diff={r_diff:.2e}  "
              f"steps_diff={s_diff:.2e}  success_same={succ_same}")
