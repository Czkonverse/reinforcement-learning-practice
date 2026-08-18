import os
import numpy as np
from envs.grid_world import GridWorld
from agents.q_learning_agent import QLearningAgent
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate(env, agent, max_step):
    state = env.reset()

    total_rewards = 0
    for step in range(max_step):

        action = agent.select_action(state, training=False)
        next_state, reward, done = env.step(action)

        total_rewards += reward

        if done:
            return done, step + 1, total_rewards

        state = next_state

    return False, max_step, total_rewards


if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    max_step = 80
    num_episodes = 100
    eval_interval = 2

    env = GridWorld()
    state_shape = (env.rows, env.cols)
    num_actions = env.num_acitons
    agent = QLearningAgent(state_shape, num_actions, alpha, gamma, epsilon)

    # stats
    episode_rewards = []
    episode_steps = []
    episode_success = []

    episode_rewards_eval = []
    episode_steps_eval = []
    episode_success_eval = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        for step in range(max_step):
            row, col = state

            action = agent.select_action(state)

            # execute action
            next_state, reward, done = env.step(action)
            total_reward += reward

            agent.update(state, action, reward, next_state, done)

            if done:
                episode_steps.append(step + 1)
                break

            state = next_state

        episode_rewards.append(total_reward)
        if not done:
            episode_steps.append(max_step)
        episode_success.append(done)

        test_env = GridWorld()
        if (episode + 1) % eval_interval == 0:
            eval_done, eval_steps, eval_total_rewards = evaluate(
                test_env, agent, max_step
            )
            episode_rewards_eval.append(eval_total_rewards)
            episode_steps_eval.append(eval_steps)
            episode_success_eval.append(eval_done)

    print(f"训练成功率: {np.mean(episode_success):.2f}")
    print(
        f"最终评估 done={episode_success_eval[-1]}, "
        f"steps={episode_steps_eval[-1]}, reward={episode_rewards_eval[-1]}"
    )

    print(episode_rewards_eval)
    print(episode_steps_eval)
    print(episode_success_eval)

    # ==================== Seaborn 绘图 ====================
    episodes = np.arange(1, num_episodes + 1)
    eval_episodes = np.arange(eval_interval, num_episodes + 1, eval_interval)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) 训练 rewards
    sns.lineplot(
        x=episodes, y=episode_rewards, ax=axes[0], color="tab:blue", label="train"
    )
    # 平滑一些训练数据会更好看，这里用简单的移动平均(窗口5)
    window = 5
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        sns.lineplot(
            x=episodes[window - 1 :],
            y=smoothed,
            ax=axes[0],
            color="tab:red",
            label=f"train (MA{window})",
        )
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Training Rewards per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()

    # 2) 训练 steps（到达目标或打满 max_step）
    sns.lineplot(x=episodes, y=episode_steps, ax=axes[1], color="tab:blue")
    axes[1].axhline(
        max_step,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"max_step={max_step}",
    )
    axes[1].set_title("Steps per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].legend()

    # 3) 训练 success
    sns.lineplot(
        x=episodes, y=episode_success, ax=axes[2], color="tab:blue", label="train"
    )
    # 若训练 success 是 bool 列表，画成 0/1 可能需要向下取整；这里直接用
    axes[2].set_title("Training Success per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Success")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # ==================== 评估结果绘图 ====================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    sns.lineplot(
        x=eval_episodes,
        y=episode_rewards_eval,
        ax=axes2[0],
        color="tab:orange",
        marker="o",
    )
    axes2[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes2[0].set_title("Evaluation Rewards")
    axes2[0].set_xlabel("Episode")
    axes2[0].set_ylabel("Total Reward")

    sns.lineplot(
        x=eval_episodes,
        y=episode_steps_eval,
        ax=axes2[1],
        color="tab:orange",
        marker="o",
    )
    axes2[1].axhline(
        max_step,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"max_step={max_step}",
    )
    axes2[1].set_title("Evaluation Steps")
    axes2[1].set_xlabel("Episode")
    axes2[1].set_ylabel("Steps")
    axes2[1].legend()

    sns.lineplot(
        x=eval_episodes,
        y=episode_success_eval,
        ax=axes2[2],
        color="tab:orange",
        marker="o",
    )
    axes2[2].set_title("Evaluation Success")
    axes2[2].set_xlabel("Episode")
    axes2[2].set_ylabel("Success")
    axes2[2].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()
