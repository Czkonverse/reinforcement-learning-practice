import os

import numpy as np
import matplotlib.pyplot as plt


def moving_average_2d(data, window):
    """
    对每个 seed 沿 episode 方向计算 moving average。

    input:
        data.shape = (num_seeds, num_episodes)

    output:
        shape = (
            num_seeds,
            num_episodes - window + 1
        )
    """
    kernel = np.ones(window) / window

    return np.array(
        [np.convolve(seed_data, kernel, mode="valid") for seed_data in data]
    )


def mean_and_std_across_seeds(data):
    """
    对所有 seeds 在同一个 episode 位置上计算 mean 和 std。

    input:
        data.shape = (num_seeds, num_points)

    output:
        mean.shape = (num_points,)
        std.shape = (num_points,)
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

    resource_dir = os.path.join(
        base_dir,
        "resource",
    )

    # --------------------------------
    # 2. Create subplots
    # --------------------------------
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    axes = axes.ravel()

    # --------------------------------
    # 3. Loop over epsilon
    # --------------------------------
    for ax, epsilon in zip(axes, epsilons):

        file_path = os.path.join(
            resource_dir,
            f"q_learning_vs_sarsa_seeds-{epsilon}.npz",
        )

        data = np.load(file_path)

        # shape: (20, 300)
        q_cliff_falls = data["q_cliff_falls"]
        sarsa_cliff_falls = data["sarsa_cliff_falls"]

        print(f"epsilon = {epsilon}")
        print(
            "  Q-learning cliff falls:",
            q_cliff_falls.shape,
        )
        print(
            "  SARSA cliff falls:",
            sarsa_cliff_falls.shape,
        )

        num_seeds, num_episodes = q_cliff_falls.shape

        # --------------------------------
        # 4. Moving average
        # --------------------------------
        q_cliff_ma = moving_average_2d(
            q_cliff_falls,
            window,
        )

        sarsa_cliff_ma = moving_average_2d(
            sarsa_cliff_falls,
            window,
        )

        # --------------------------------
        # 5. Across-seed mean ± std
        # --------------------------------
        q_mean, q_std = mean_and_std_across_seeds(q_cliff_ma)

        sarsa_mean, sarsa_std = mean_and_std_across_seeds(sarsa_cliff_ma)

        # moving average 第一个点对应 episode = window
        episodes = np.arange(
            window,
            num_episodes + 1,
        )

        # --------------------------------
        # 6. Q-learning
        # --------------------------------
        ax.plot(
            episodes,
            q_mean,
            linewidth=2.5,
            label="Q-learning",
        )

        ax.fill_between(
            episodes,
            # Cliff Falls 不可能小于 0
            np.maximum(
                q_mean - q_std,
                0,
            ),
            q_mean + q_std,
            alpha=0.2,
        )

        # --------------------------------
        # 7. SARSA
        # --------------------------------
        ax.plot(
            episodes,
            sarsa_mean,
            linewidth=2.5,
            label="SARSA",
        )

        ax.fill_between(
            episodes,
            np.maximum(
                sarsa_mean - sarsa_std,
                0,
            ),
            sarsa_mean + sarsa_std,
            alpha=0.2,
        )

        # --------------------------------
        # 8. Style
        # --------------------------------
        ax.set_title(f"Epsilon = {epsilon}")

        ax.set_xlabel("Episode")

        ax.set_ylabel("Cliff Falls per Episode")

        ax.set_ylim(bottom=0)

        ax.grid(alpha=0.3)

        ax.legend()

    # --------------------------------
    # 9. Final layout
    # --------------------------------
    fig.suptitle(
        ("Q-learning vs SARSA: " f"Cliff Falls by Epsilon (MA={window})"),
        fontsize=16,
    )

    plt.tight_layout()

    plt.show()
