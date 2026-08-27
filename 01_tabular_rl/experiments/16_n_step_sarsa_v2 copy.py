import numpy as np

from agents.n_step_sarsa_agent_v2 import NStepSarsaAgent
from envs.cliff_world import CliffWorld


def evaluate(env, agent, max_step):
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
    num_episodes = 200
    max_step = 200
    n = 3
    epsilon = 0.2
    gamma = 0.9
    alpha = 0.1

    eval_every = 20

    np.random.seed(0)

    env = CliffWorld()
    agent = NStepSarsaAgent(
        state_shape=(env.rows, env.cols),
        num_actions=env.num_actions,
        n=n,
        alpha=0.1,
        gamma=0.9,
        epsilon=epsilon,
    )

    episode_rewards = []
    episode_steps = []
    episode_success = []
    eval_rewards = []
    eval_success = []

    for episode in range(num_episodes):
        state = env.reset()
        action = agent.begin_episode(action)

        total_reward = 0.0
        done = False
        for step in range(max_step):
            next_state, reward, done = env.step(action)
            total_reward += reward

            action = agent.learn(reward, next_state, done)

            if done:
                break

        # out of max_step limitation
        if not done:
            agent.end_episode()

        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)
        episode_success.append(done)

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
