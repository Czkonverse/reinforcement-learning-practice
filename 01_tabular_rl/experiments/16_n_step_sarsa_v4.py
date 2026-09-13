"""n 步 SARSA 测试脚本：仿照 16_n_step_sarsa_v3.py 的流程，
用 CliffWorld 训练并贪心评估 agents/n_step_sarsa_agent.py 中的 NStepSarsaAgent。

和 v3 的差异（因为被测 agent 的接口不同）：
- 对外接口是 begin_episode(state) / step(reward, next_state, terminal, truncated)
  / end_episode(reason)；
- agent.step 需要显式区分「真正终止 terminal」和「步数上限截断 truncated」：
  * terminal=True 时 step 返回 None（终止状态后面没有动作）；
  * truncated=True 时 step 返回下一步动作，但把尾部 transition 留给
    end_episode("truncated") 用 bootstrap 补完；
- episode 结束后训练循环必须调用 end_episode("terminal") 或
  end_episode("truncated") 来 flush 缓冲区并清空。
"""

import numpy as np

from agents.n_step_sarsa_agent import NStepSarsaAgent
from envs.cliff_world import CliffWorld


def evaluate(env, agent, max_step):
    """用贪心策略评估一个 episode，返回 (是否到达终点, 步数, 累计回报)。"""
    state = env.reset()
    total_reward = 0.0
    for step in range(max_step):
        action = agent.select_action(state, training=False)
        next_state, reward, done = env.step(action)
        total_reward += reward
        if done:
            return True, step + 1, total_reward
        state = next_state
    return False, max_step, total_reward


if __name__ == "__main__":
    # 超参数（与 v3 保持一致）
    num_episodes = 200
    max_step = 200
    n = 3
    epsilon = 0.2
    gamma = 0.9
    alpha = 0.1
    eval_every = 20
    seed = 0
    np.random.seed(seed)

    env = CliffWorld()
    agent = NStepSarsaAgent(
        (env.rows, env.cols),
        env.num_actions,
        n=n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        seed=seed,
    )

    episode_rewards = []
    episode_steps = []
    episode_success = []
    eval_rewards = []
    eval_success = []

    for episode in range(num_episodes):
        state = env.reset()
        action = agent.begin_episode(state)

        total_reward = 0.0
        terminal = False
        truncated = False

        for step in range(max_step):
            next_state, reward, done = env.step(action)
            total_reward += reward

            # 区分真正终止与步数上限截断
            terminal = done
            truncated = (not done) and (step == max_step - 1)

            # agent 内部完成：缓存 transition、n 步更新；
            # terminal/truncated 时把尾部 flush 交给 end_episode
            action = agent.step(reward, next_state, terminal, truncated)

            if done:
                break

        # episode 结束：真正终止用 "terminal"，步数上限截断用 "truncated"
        agent.end_episode("terminal" if terminal else "truncated")

        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)
        episode_success.append(terminal)

        if (episode + 1) % eval_every == 0:
            eval_done, _, eval_reward = evaluate(env, agent, max_step)
            eval_success.append(eval_done)
            eval_rewards.append(eval_reward)

    print("===== n-step SARSA (n=%d) 训练统计 =====" % n)
    print("avg episode steps :", np.mean(episode_steps))
    print("avg episode reward:", np.mean(episode_rewards))
    print("success rate      :", np.mean(episode_success))

    # 周期性贪心评估
    print("greedy eval rewards:", [round(r, 1) for r in eval_rewards])
    print("greedy eval success:", eval_success)

    # 最终贪心评估
    eval_done, eval_step, eval_reward = evaluate(env, agent, max_step)
    print(
        "final greedy eval -> done:",
        eval_done,
        "| steps:",
        eval_step,
        "| reward:",
        eval_reward,
    )
