import os

import numpy as np
import matplotlib.pyplot as plt


def moving_average_2d(data, window):
    """
    对每一个 seed 分别沿 episode 方向做 moving average。

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
    对同一个 episode/window 位置上的所有 seeds 求 mean ± std。

    input:
        data.shape = (num_seeds, num_points)

    output:
        mean.shape = (num_points,)
        std.shape = (num_points,)
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)

    return mean, std


def prepare_curve(data, window):
    """
    moving average
        ↓
    across-seed mean ± std
    """
    ma = moving_average_2d(data, window)

    mean, std = mean_and_std_across_seeds(ma)

    return mean, std


def summarize_last_episodes(data, last_n):
    """
    对每一个 seed 的最后 last_n 个 episodes 先求平均，
    再统计 20 个 seeds 的 mean ± std。

    input:
        data.shape = (20, 300)
    """

    # 每个 seed 一个后期平均值
    seed_means = np.mean(
        data[:, -last_n:],
        axis=1,
    )

    mean = np.mean(seed_means)
    std = np.std(seed_means, ddof=1)

    return mean, std


if __name__ == "__main__":

    # ============================================================
    # 1. Config
    # ============================================================

    window = 20

    # 用于打印“训练后期”的统计
    last_n = 50

    fixed_epsilon = 0.2

    base_dir = os.path.dirname(os.path.abspath(__file__))

    resource_dir = os.path.join(
        base_dir,
        "resource",
    )

    fixed_path = os.path.join(
        resource_dir,
        "q_learning_vs_sarsa_seeds-0.2.npz",
    )

    decay_path = os.path.join(
        resource_dir,
        "q_learning_vs_sarsa_linear_decay.npz",
    )

    # ============================================================
    # 2. Load data
    # ============================================================

    fixed_data = np.load(fixed_path)
    decay_data = np.load(decay_path)

    # ---------- Fixed epsilon ----------

    q_fixed_rewards = fixed_data["q_rewards"]
    q_fixed_steps = fixed_data["q_steps"]
    q_fixed_success = fixed_data["q_success"]
    q_fixed_cliff = fixed_data["q_cliff_falls"]

    sarsa_fixed_rewards = fixed_data["sarsa_rewards"]
    sarsa_fixed_steps = fixed_data["sarsa_steps"]
    sarsa_fixed_success = fixed_data["sarsa_success"]
    sarsa_fixed_cliff = fixed_data["sarsa_cliff_falls"]

    # ---------- Linear decay ----------

    q_decay_rewards = decay_data["q_rewards"]
    q_decay_steps = decay_data["q_steps"]
    q_decay_success = decay_data["q_success"]
    q_decay_cliff = decay_data["q_cliff_falls"]

    sarsa_decay_rewards = decay_data["sarsa_rewards"]
    sarsa_decay_steps = decay_data["sarsa_steps"]
    sarsa_decay_success = decay_data["sarsa_success"]
    sarsa_decay_cliff = decay_data["sarsa_cliff_falls"]

    # decay 文件中保存的是 (20, 300)
    q_decay_epsilons = decay_data["q_epsilons"]

    # ============================================================
    # 3. Sanity checks
    # ============================================================

    print("Fixed Q rewards:", q_fixed_rewards.shape)
    print("Decay Q rewards:", q_decay_rewards.shape)

    print("Fixed SARSA rewards:", sarsa_fixed_rewards.shape)
    print("Decay SARSA rewards:", sarsa_decay_rewards.shape)

    if q_fixed_rewards.shape != q_decay_rewards.shape:
        raise ValueError("Fixed and decay experiments have different shapes.")

    if sarsa_fixed_rewards.shape != sarsa_decay_rewards.shape:
        raise ValueError("Fixed and decay SARSA experiments have different shapes.")

    num_seeds, num_episodes = q_fixed_rewards.shape

    print()
    print("num_seeds:", num_seeds)
    print("num_episodes:", num_episodes)

    # ============================================================
    # 4. Epsilon schedules
    # ============================================================

    # Fixed epsilon
    fixed_epsilon_history = np.full(
        num_episodes,
        fixed_epsilon,
    )

    # 所有 seed 的 decay epsilon 都相同，
    # 所以取 seed 0 即可
    decay_epsilon_history = q_decay_epsilons[0]

    episode_raw = np.arange(
        1,
        num_episodes + 1,
    )

    # ============================================================
    # 5. Prepare reward curves
    # ============================================================

    q_fixed_reward_mean, q_fixed_reward_std = prepare_curve(
        q_fixed_rewards,
        window,
    )

    q_decay_reward_mean, q_decay_reward_std = prepare_curve(
        q_decay_rewards,
        window,
    )

    sarsa_fixed_reward_mean, sarsa_fixed_reward_std = prepare_curve(
        sarsa_fixed_rewards,
        window,
    )

    sarsa_decay_reward_mean, sarsa_decay_reward_std = prepare_curve(
        sarsa_decay_rewards,
        window,
    )

    # ============================================================
    # 6. Prepare cliff-fall curves
    # ============================================================

    q_fixed_cliff_mean, q_fixed_cliff_std = prepare_curve(
        q_fixed_cliff,
        window,
    )

    q_decay_cliff_mean, q_decay_cliff_std = prepare_curve(
        q_decay_cliff,
        window,
    )

    sarsa_fixed_cliff_mean, sarsa_fixed_cliff_std = prepare_curve(
        sarsa_fixed_cliff,
        window,
    )

    sarsa_decay_cliff_mean, sarsa_decay_cliff_std = prepare_curve(
        sarsa_decay_cliff,
        window,
    )

    # Moving average 的第一个点对应 episode = window
    episode_ma = np.arange(
        window,
        num_episodes + 1,
    )

    # ============================================================
    # 7. Print late-training summary
    # ============================================================

    print()
    print("=" * 70)
    print(f"Late-training summary: last {last_n} episodes")
    print("=" * 70)

    # ---------- Reward ----------

    q_fixed_reward_late = summarize_last_episodes(
        q_fixed_rewards,
        last_n,
    )

    q_decay_reward_late = summarize_last_episodes(
        q_decay_rewards,
        last_n,
    )

    sarsa_fixed_reward_late = summarize_last_episodes(
        sarsa_fixed_rewards,
        last_n,
    )

    sarsa_decay_reward_late = summarize_last_episodes(
        sarsa_decay_rewards,
        last_n,
    )

    print("\nReward")

    print(
        "Q-learning fixed : "
        f"{q_fixed_reward_late[0]:.2f} "
        f"± {q_fixed_reward_late[1]:.2f}"
    )

    print(
        "Q-learning decay : "
        f"{q_decay_reward_late[0]:.2f} "
        f"± {q_decay_reward_late[1]:.2f}"
    )

    print(
        "SARSA fixed      : "
        f"{sarsa_fixed_reward_late[0]:.2f} "
        f"± {sarsa_fixed_reward_late[1]:.2f}"
    )

    print(
        "SARSA decay      : "
        f"{sarsa_decay_reward_late[0]:.2f} "
        f"± {sarsa_decay_reward_late[1]:.2f}"
    )

    # ---------- Cliff falls ----------

    q_fixed_cliff_late = summarize_last_episodes(
        q_fixed_cliff,
        last_n,
    )

    q_decay_cliff_late = summarize_last_episodes(
        q_decay_cliff,
        last_n,
    )

    sarsa_fixed_cliff_late = summarize_last_episodes(
        sarsa_fixed_cliff,
        last_n,
    )

    sarsa_decay_cliff_late = summarize_last_episodes(
        sarsa_decay_cliff,
        last_n,
    )

    print("\nCliff falls per episode")

    print(
        "Q-learning fixed : "
        f"{q_fixed_cliff_late[0]:.3f} "
        f"± {q_fixed_cliff_late[1]:.3f}"
    )

    print(
        "Q-learning decay : "
        f"{q_decay_cliff_late[0]:.3f} "
        f"± {q_decay_cliff_late[1]:.3f}"
    )

    print(
        "SARSA fixed      : "
        f"{sarsa_fixed_cliff_late[0]:.3f} "
        f"± {sarsa_fixed_cliff_late[1]:.3f}"
    )

    print(
        "SARSA decay      : "
        f"{sarsa_decay_cliff_late[0]:.3f} "
        f"± {sarsa_decay_cliff_late[1]:.3f}"
    )

    # ---------- Steps ----------

    print("\nEpisode steps")

    for name, array in [
        ("Q-learning fixed", q_fixed_steps),
        ("Q-learning decay", q_decay_steps),
        ("SARSA fixed", sarsa_fixed_steps),
        ("SARSA decay", sarsa_decay_steps),
    ]:
        mean, std = summarize_last_episodes(
            array,
            last_n,
        )

        print(f"{name:17s}: " f"{mean:.2f} ± {std:.2f}")

    # ============================================================
    # 8. Figure 1: epsilon schedule
    # ============================================================

    fig_epsilon, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        episode_raw,
        fixed_epsilon_history,
        linewidth=2.5,
        label="Fixed epsilon = 0.2",
    )

    ax.plot(
        episode_raw,
        decay_epsilon_history,
        linewidth=2.5,
        label="Linear decay: 0.2 -> 0.01",
    )

    ax.set_title("Fixed Epsilon vs Linear Epsilon Decay")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")

    ax.set_ylim(
        0,
        fixed_epsilon + 0.02,
    )

    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # ============================================================
    # 9. Figure 2: Reward + Cliff Falls
    # ============================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        sharex=True,
    )

    (
        ax_q_reward,
        ax_sarsa_reward,
        ax_q_cliff,
        ax_sarsa_cliff,
    ) = axes.ravel()

    # ============================================================
    # 9.1 Q-learning Reward
    # ============================================================

    ax_q_reward.plot(
        episode_ma,
        q_fixed_reward_mean,
        linewidth=2.5,
        linestyle="--",
        label="Fixed ε=0.2",
    )

    ax_q_reward.fill_between(
        episode_ma,
        q_fixed_reward_mean - q_fixed_reward_std,
        q_fixed_reward_mean + q_fixed_reward_std,
        alpha=0.15,
    )

    ax_q_reward.plot(
        episode_ma,
        q_decay_reward_mean,
        linewidth=2.5,
        label="Linear decay",
    )

    ax_q_reward.fill_between(
        episode_ma,
        q_decay_reward_mean - q_decay_reward_std,
        q_decay_reward_mean + q_decay_reward_std,
        alpha=0.15,
    )

    ax_q_reward.set_title("Q-learning: Training Reward")

    ax_q_reward.set_ylabel("Episode Reward")

    ax_q_reward.grid(alpha=0.3)
    ax_q_reward.legend()

    # ============================================================
    # 9.2 SARSA Reward
    # ============================================================

    ax_sarsa_reward.plot(
        episode_ma,
        sarsa_fixed_reward_mean,
        linewidth=2.5,
        linestyle="--",
        label="Fixed ε=0.2",
    )

    ax_sarsa_reward.fill_between(
        episode_ma,
        sarsa_fixed_reward_mean - sarsa_fixed_reward_std,
        sarsa_fixed_reward_mean + sarsa_fixed_reward_std,
        alpha=0.15,
    )

    ax_sarsa_reward.plot(
        episode_ma,
        sarsa_decay_reward_mean,
        linewidth=2.5,
        label="Linear decay",
    )

    ax_sarsa_reward.fill_between(
        episode_ma,
        sarsa_decay_reward_mean - sarsa_decay_reward_std,
        sarsa_decay_reward_mean + sarsa_decay_reward_std,
        alpha=0.15,
    )

    ax_sarsa_reward.set_title("SARSA: Training Reward")

    ax_sarsa_reward.set_ylabel("Episode Reward")

    ax_sarsa_reward.grid(alpha=0.3)
    ax_sarsa_reward.legend()

    # ============================================================
    # 9.3 Q-learning Cliff Falls
    # ============================================================

    ax_q_cliff.plot(
        episode_ma,
        q_fixed_cliff_mean,
        linewidth=2.5,
        linestyle="--",
        label="Fixed ε=0.2",
    )

    ax_q_cliff.fill_between(
        episode_ma,
        np.maximum(
            q_fixed_cliff_mean - q_fixed_cliff_std,
            0,
        ),
        q_fixed_cliff_mean + q_fixed_cliff_std,
        alpha=0.15,
    )

    ax_q_cliff.plot(
        episode_ma,
        q_decay_cliff_mean,
        linewidth=2.5,
        label="Linear decay",
    )

    ax_q_cliff.fill_between(
        episode_ma,
        np.maximum(
            q_decay_cliff_mean - q_decay_cliff_std,
            0,
        ),
        q_decay_cliff_mean + q_decay_cliff_std,
        alpha=0.15,
    )

    ax_q_cliff.set_title("Q-learning: Cliff Falls")

    ax_q_cliff.set_xlabel("Episode")

    ax_q_cliff.set_ylabel("Cliff Falls per Episode")

    ax_q_cliff.set_ylim(bottom=0)

    ax_q_cliff.grid(alpha=0.3)
    ax_q_cliff.legend()

    # ============================================================
    # 9.4 SARSA Cliff Falls
    # ============================================================

    ax_sarsa_cliff.plot(
        episode_ma,
        sarsa_fixed_cliff_mean,
        linewidth=2.5,
        linestyle="--",
        label="Fixed ε=0.2",
    )

    ax_sarsa_cliff.fill_between(
        episode_ma,
        np.maximum(
            sarsa_fixed_cliff_mean - sarsa_fixed_cliff_std,
            0,
        ),
        sarsa_fixed_cliff_mean + sarsa_fixed_cliff_std,
        alpha=0.15,
    )

    ax_sarsa_cliff.plot(
        episode_ma,
        sarsa_decay_cliff_mean,
        linewidth=2.5,
        label="Linear decay",
    )

    ax_sarsa_cliff.fill_between(
        episode_ma,
        np.maximum(
            sarsa_decay_cliff_mean - sarsa_decay_cliff_std,
            0,
        ),
        sarsa_decay_cliff_mean + sarsa_decay_cliff_std,
        alpha=0.15,
    )

    ax_sarsa_cliff.set_title("SARSA: Cliff Falls")

    ax_sarsa_cliff.set_xlabel("Episode")

    ax_sarsa_cliff.set_ylabel("Cliff Falls per Episode")

    ax_sarsa_cliff.set_ylim(bottom=0)

    ax_sarsa_cliff.grid(alpha=0.3)
    ax_sarsa_cliff.legend()

    fig.suptitle(
        ("Fixed ε=0.2 vs Linear Decay 0.2→0.01 " f"(20 Seeds, MA={window})"),
        fontsize=16,
    )

    plt.tight_layout()

    # ============================================================
    # 10. Show
    # ============================================================

    plt.show()
