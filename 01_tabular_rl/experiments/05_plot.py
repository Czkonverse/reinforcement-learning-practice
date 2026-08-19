import os

import numpy as np
import matplotlib.pyplot as plt


def moving_average_2d(data, window):
    """
    对 shape = (num_seeds, num_episodes) 的数据，
    分别对每一个 seed 沿 episode 方向计算 moving average。

    input:
        data.shape = (20, 300)

    output:
        如果 window = 20
        shape = (20, 281)
    """
    kernel = np.ones(window) / window

    return np.array(
        [np.convolve(seed_data, kernel, mode="valid") for seed_data in data]
    )


def mean_and_std(data):
    """
    对多个 seeds 求平均值和标准差。

    data.shape:
        (num_seeds, num_points)

    axis=0:
        对同一个 episode/window 上的所有 seeds 求统计量。
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)

    return mean, std


if __name__ == "__main__":

    ########################
    # 1. Load experiment data
    ########################

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "resource",
        "q_learning_vs_sarsa_seeds.npz",
    )

    data = np.load(data_path)

    # Q-learning
    q_rewards = data["q_rewards"]
    q_steps = data["q_steps"]
    q_success = data["q_success"]
    q_cliff_falls = data["q_cliff_falls"]

    # SARSA
    sarsa_rewards = data["sarsa_rewards"]
    sarsa_steps = data["sarsa_steps"]
    sarsa_success = data["sarsa_success"]
    sarsa_cliff_falls = data["sarsa_cliff_falls"]

    print("Q-learning rewards shape:", q_rewards.shape)
    print("SARSA rewards shape:", sarsa_rewards.shape)

    num_seeds, num_episodes = q_rewards.shape

    ########################
    # 2. Moving average
    ########################

    window = 20

    # reward
    q_reward_ma = moving_average_2d(q_rewards, window)
    sarsa_reward_ma = moving_average_2d(sarsa_rewards, window)

    # cliff falls
    q_cliff_ma = moving_average_2d(q_cliff_falls, window)
    sarsa_cliff_ma = moving_average_2d(
        sarsa_cliff_falls,
        window,
    )

    # steps
    q_steps_ma = moving_average_2d(q_steps, window)
    sarsa_steps_ma = moving_average_2d(sarsa_steps, window)

    # success:
    # True / False -> 1 / 0
    # moving average 就是 rolling success rate
    q_success_ma = moving_average_2d(
        q_success.astype(float),
        window,
    )

    sarsa_success_ma = moving_average_2d(
        sarsa_success.astype(float),
        window,
    )

    ########################
    # 3. Across-seed statistics
    ########################

    # Reward
    q_reward_mean, q_reward_std = mean_and_std(q_reward_ma)
    sarsa_reward_mean, sarsa_reward_std = mean_and_std(sarsa_reward_ma)

    # Cliff falls
    q_cliff_mean, q_cliff_std = mean_and_std(q_cliff_ma)
    sarsa_cliff_mean, sarsa_cliff_std = mean_and_std(sarsa_cliff_ma)

    # Steps
    q_steps_mean, q_steps_std = mean_and_std(q_steps_ma)
    sarsa_steps_mean, sarsa_steps_std = mean_and_std(sarsa_steps_ma)

    # Success rate
    q_success_mean, q_success_std = mean_and_std(q_success_ma)
    sarsa_success_mean, sarsa_success_std = mean_and_std(sarsa_success_ma)

    ########################
    # 4. X-axis
    ########################

    # moving average 的第一个点对应 episode = window
    episodes = np.arange(window, num_episodes + 1)

    ########################
    # 5. Plot
    ########################

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    ax_reward, ax_cliff, ax_steps, ax_success = axes.ravel()

    ########################
    # 5.1 Training Reward
    ########################

    ax_reward.plot(
        episodes,
        q_reward_mean,
        linewidth=2.5,
        label="Q-learning",
    )

    ax_reward.fill_between(
        episodes,
        q_reward_mean - q_reward_std,
        q_reward_mean + q_reward_std,
        alpha=0.2,
    )

    ax_reward.plot(
        episodes,
        sarsa_reward_mean,
        linewidth=2.5,
        label="SARSA",
    )

    ax_reward.fill_between(
        episodes,
        sarsa_reward_mean - sarsa_reward_std,
        sarsa_reward_mean + sarsa_reward_std,
        alpha=0.2,
    )

    ax_reward.set_title(f"Training Reward ({num_seeds} Seeds, MA={window})")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Episode Reward")
    ax_reward.legend()
    ax_reward.grid(alpha=0.3)

    ########################
    # 5.2 Cliff Falls
    ########################

    ax_cliff.plot(
        episodes,
        q_cliff_mean,
        linewidth=2.5,
        label="Q-learning",
    )

    # cliff falls 不可能小于 0
    ax_cliff.fill_between(
        episodes,
        np.maximum(q_cliff_mean - q_cliff_std, 0),
        q_cliff_mean + q_cliff_std,
        alpha=0.2,
    )

    ax_cliff.plot(
        episodes,
        sarsa_cliff_mean,
        linewidth=2.5,
        label="SARSA",
    )

    ax_cliff.fill_between(
        episodes,
        np.maximum(sarsa_cliff_mean - sarsa_cliff_std, 0),
        sarsa_cliff_mean + sarsa_cliff_std,
        alpha=0.2,
    )

    ax_cliff.set_title(f"Cliff Falls ({num_seeds} Seeds, MA={window})")
    ax_cliff.set_xlabel("Episode")
    ax_cliff.set_ylabel("Cliff Falls per Episode")
    ax_cliff.legend()
    ax_cliff.grid(alpha=0.3)

    ########################
    # 5.3 Episode Steps
    ########################

    ax_steps.plot(
        episodes,
        q_steps_mean,
        linewidth=2.5,
        label="Q-learning",
    )

    ax_steps.fill_between(
        episodes,
        np.maximum(q_steps_mean - q_steps_std, 0),
        q_steps_mean + q_steps_std,
        alpha=0.2,
    )

    ax_steps.plot(
        episodes,
        sarsa_steps_mean,
        linewidth=2.5,
        label="SARSA",
    )

    ax_steps.fill_between(
        episodes,
        np.maximum(sarsa_steps_mean - sarsa_steps_std, 0),
        sarsa_steps_mean + sarsa_steps_std,
        alpha=0.2,
    )

    ax_steps.set_title(f"Episode Steps ({num_seeds} Seeds, MA={window})")
    ax_steps.set_xlabel("Episode")
    ax_steps.set_ylabel("Episode Steps")
    ax_steps.legend()
    ax_steps.grid(alpha=0.3)

    ########################
    # 5.4 Rolling Success Rate
    ########################

    ax_success.plot(
        episodes,
        q_success_mean,
        linewidth=2.5,
        label="Q-learning",
    )

    ax_success.fill_between(
        episodes,
        np.clip(
            q_success_mean - q_success_std,
            0,
            1,
        ),
        np.clip(
            q_success_mean + q_success_std,
            0,
            1,
        ),
        alpha=0.2,
    )

    ax_success.plot(
        episodes,
        sarsa_success_mean,
        linewidth=2.5,
        label="SARSA",
    )

    ax_success.fill_between(
        episodes,
        np.clip(
            sarsa_success_mean - sarsa_success_std,
            0,
            1,
        ),
        np.clip(
            sarsa_success_mean + sarsa_success_std,
            0,
            1,
        ),
        alpha=0.2,
    )

    ax_success.set_title(f"Rolling Success Rate ({num_seeds} Seeds, Window={window})")
    ax_success.set_xlabel("Episode")
    ax_success.set_ylabel("Success Rate")
    ax_success.set_ylim(0, 1.05)
    ax_success.legend()
    ax_success.grid(alpha=0.3)

    ########################
    # 6. Final layout
    ########################

    fig.suptitle(
        "Q-learning vs SARSA: Multi-seed Training Performance",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()
