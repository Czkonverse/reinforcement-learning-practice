import os

import numpy as np
import matplotlib.pyplot as plt


def moving_average_2d(data, window):
    """
    data shape: (num_seeds, num_episodes)

    对每个 seed 分别沿着 episode 方向做 moving average
    返回 shape: (num_seeds, num_episodes - window + 1)
    """
    kernel = np.ones(window) / window

    return np.array(
        [np.convolve(seed_data, kernel, mode="valid") for seed_data in data]
    )


def mean_and_std_across_seeds(data):
    """
    data shape: (num_seeds, num_points)

    对所有 seeds 求 mean 和 std
    返回:
        mean shape: (num_points,)
        std  shape: (num_points,)
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    return mean, std


if __name__ == "__main__":
    # --------------------------------
    # 1. Config
    # --------------------------------
    epsilons = [0.05, 0.1, 0.2, 0.3]
    window = 20

    base_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(base_dir, "resource")

    # --------------------------------
    # 2. Create subplots
    # --------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # --------------------------------
    # 3. Loop over epsilon files
    # --------------------------------
    for ax, epsilon in zip(axes, epsilons):
        file_path = os.path.join(
            resource_dir,
            f"q_learning_vs_sarsa_seeds-{epsilon}.npz",
        )

        data = np.load(file_path)

        q_rewards = data["q_rewards"]  # shape: (20, 300)
        sarsa_rewards = data["sarsa_rewards"]  # shape: (20, 300)

        print(f"epsilon={epsilon}")
        print("  q_rewards shape:", q_rewards.shape)
        print("  sarsa_rewards shape:", sarsa_rewards.shape)

        num_seeds, num_episodes = q_rewards.shape

        # --------------------------------
        # 4. Moving average for each seed
        # --------------------------------
        q_reward_ma = moving_average_2d(q_rewards, window)
        sarsa_reward_ma = moving_average_2d(sarsa_rewards, window)

        # --------------------------------
        # 5. Across-seed mean ± std
        # --------------------------------
        q_mean, q_std = mean_and_std_across_seeds(q_reward_ma)
        sarsa_mean, sarsa_std = mean_and_std_across_seeds(sarsa_reward_ma)

        # x-axis
        episodes = np.arange(window, num_episodes + 1)

        # --------------------------------
        # 6. Plot Q-learning
        # --------------------------------
        ax.plot(
            episodes,
            q_mean,
            linewidth=2.5,
            label="Q-learning",
        )

        ax.fill_between(
            episodes,
            q_mean - q_std,
            q_mean + q_std,
            alpha=0.2,
        )

        # --------------------------------
        # 7. Plot SARSA
        # --------------------------------
        ax.plot(
            episodes,
            sarsa_mean,
            linewidth=2.5,
            label="SARSA",
        )

        ax.fill_between(
            episodes,
            sarsa_mean - sarsa_std,
            sarsa_mean + sarsa_std,
            alpha=0.2,
        )

        # --------------------------------
        # 8. Style
        # --------------------------------
        ax.set_title(f"Epsilon = {epsilon}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Training Reward")
        ax.grid(alpha=0.3)
        ax.legend()

    # --------------------------------
    # 9. Final layout
    # --------------------------------
    fig.suptitle(
        f"Q-learning vs SARSA: Training Reward Curves by Epsilon (MA={window})",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()
