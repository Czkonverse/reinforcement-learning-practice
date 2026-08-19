import os

import numpy as np
import matplotlib.pyplot as plt


def summarize_last_window(data, tail_window=100):
    """
    data shape: (num_seeds, num_episodes)

    先对每个 seed 的最后 tail_window 个 episode 求平均，
    再对所有 seeds 求 mean 和 std。

    return:
        seed_means: shape (num_seeds,)
        mean_value: scalar
        std_value: scalar
    """
    seed_means = np.mean(data[:, -tail_window:], axis=1)
    mean_value = np.mean(seed_means)
    std_value = np.std(seed_means, ddof=1)
    return seed_means, mean_value, std_value


if __name__ == "__main__":
    # ----------------------------
    # 1. Basic config
    # ----------------------------
    epsilons = [0.05, 0.1, 0.2, 0.3]
    tail_window = 100

    base_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(base_dir, "resource")

    # ----------------------------
    # 2. Containers for summary stats
    # ----------------------------
    q_reward_means = []
    q_reward_stds = []

    q_cliff_means = []
    q_cliff_stds = []

    q_steps_means = []
    q_steps_stds = []

    q_success_means = []
    q_success_stds = []

    sarsa_reward_means = []
    sarsa_reward_stds = []

    sarsa_cliff_means = []
    sarsa_cliff_stds = []

    sarsa_steps_means = []
    sarsa_steps_stds = []

    sarsa_success_means = []
    sarsa_success_stds = []

    # ----------------------------
    # 3. Load each epsilon result and summarize
    # ----------------------------
    for epsilon in epsilons:
        file_path = os.path.join(
            resource_dir,
            f"q_learning_vs_sarsa_seeds-{epsilon}.npz",
        )

        data = np.load(file_path)

        # Q-learning
        q_rewards = data["q_rewards"]  # shape: (20, 300)
        q_steps = data["q_steps"]  # shape: (20, 300)
        q_success = data["q_success"].astype(float)  # bool -> float
        q_cliff_falls = data["q_cliff_falls"]  # shape: (20, 300)

        # SARSA
        sarsa_rewards = data["sarsa_rewards"]
        sarsa_steps = data["sarsa_steps"]
        sarsa_success = data["sarsa_success"].astype(float)
        sarsa_cliff_falls = data["sarsa_cliff_falls"]

        # ---- summarize Q-learning ----
        _, mean_value, std_value = summarize_last_window(
            q_rewards,
            tail_window,
        )
        q_reward_means.append(mean_value)
        q_reward_stds.append(std_value)

        _, mean_value, std_value = summarize_last_window(
            q_cliff_falls,
            tail_window,
        )
        q_cliff_means.append(mean_value)
        q_cliff_stds.append(std_value)

        _, mean_value, std_value = summarize_last_window(
            q_steps,
            tail_window,
        )
        q_steps_means.append(mean_value)
        q_steps_stds.append(std_value)

        _, mean_value, std_value = summarize_last_window(
            q_success,
            tail_window,
        )
        q_success_means.append(mean_value)
        q_success_stds.append(std_value)

        # ---- summarize SARSA ----
        _, mean_value, std_value = summarize_last_window(
            sarsa_rewards,
            tail_window,
        )
        sarsa_reward_means.append(mean_value)
        sarsa_reward_stds.append(std_value)

        _, mean_value, std_value = summarize_last_window(
            sarsa_cliff_falls,
            tail_window,
        )
        sarsa_cliff_means.append(mean_value)
        sarsa_cliff_stds.append(std_value)

        _, mean_value, std_value = summarize_last_window(
            sarsa_steps,
            tail_window,
        )
        sarsa_steps_means.append(mean_value)
        sarsa_steps_stds.append(std_value)

        _, mean_value, std_value = summarize_last_window(
            sarsa_success,
            tail_window,
        )
        sarsa_success_means.append(mean_value)
        sarsa_success_stds.append(std_value)

    # Convert to numpy arrays
    q_reward_means = np.array(q_reward_means)
    q_reward_stds = np.array(q_reward_stds)

    q_cliff_means = np.array(q_cliff_means)
    q_cliff_stds = np.array(q_cliff_stds)

    q_steps_means = np.array(q_steps_means)
    q_steps_stds = np.array(q_steps_stds)

    q_success_means = np.array(q_success_means)
    q_success_stds = np.array(q_success_stds)

    sarsa_reward_means = np.array(sarsa_reward_means)
    sarsa_reward_stds = np.array(sarsa_reward_stds)

    sarsa_cliff_means = np.array(sarsa_cliff_means)
    sarsa_cliff_stds = np.array(sarsa_cliff_stds)

    sarsa_steps_means = np.array(sarsa_steps_means)
    sarsa_steps_stds = np.array(sarsa_steps_stds)

    sarsa_success_means = np.array(sarsa_success_means)
    sarsa_success_stds = np.array(sarsa_success_stds)

    # ----------------------------
    # 4. Print summary in terminal
    # ----------------------------
    print(f"Summary over last {tail_window} episodes")
    print("-" * 80)
    for i, epsilon in enumerate(epsilons):
        print(f"epsilon = {epsilon}")
        print(
            f"  Q-learning | reward: {q_reward_means[i]:.2f} ± {q_reward_stds[i]:.2f}, "
            f"cliff falls: {q_cliff_means[i]:.3f} ± {q_cliff_stds[i]:.3f}, "
            f"steps: {q_steps_means[i]:.2f} ± {q_steps_stds[i]:.2f}, "
            f"success: {q_success_means[i]:.3f} ± {q_success_stds[i]:.3f}"
        )
        print(
            f"  SARSA      | reward: {sarsa_reward_means[i]:.2f} ± {sarsa_reward_stds[i]:.2f}, "
            f"cliff falls: {sarsa_cliff_means[i]:.3f} ± {sarsa_cliff_stds[i]:.3f}, "
            f"steps: {sarsa_steps_means[i]:.2f} ± {sarsa_steps_stds[i]:.2f}, "
            f"success: {sarsa_success_means[i]:.3f} ± {sarsa_success_stds[i]:.3f}"
        )
        print("-" * 80)

    # ----------------------------
    # 5. Plot
    # ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_reward, ax_cliff, ax_steps, ax_success = axes.ravel()

    # ---------- 5.1 Reward vs Epsilon ----------
    ax_reward.errorbar(
        epsilons,
        q_reward_means,
        yerr=q_reward_stds,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Q-learning",
    )
    ax_reward.errorbar(
        epsilons,
        sarsa_reward_means,
        yerr=sarsa_reward_stds,
        marker="o",
        linewidth=2,
        capsize=4,
        label="SARSA",
    )
    ax_reward.set_title(f"Average Reward (Last {tail_window} Episodes)")
    ax_reward.set_xlabel("Epsilon")
    ax_reward.set_ylabel("Episode Reward")
    ax_reward.legend()
    ax_reward.grid(alpha=0.3)

    # ---------- 5.2 Cliff Falls vs Epsilon ----------
    ax_cliff.errorbar(
        epsilons,
        q_cliff_means,
        yerr=q_cliff_stds,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Q-learning",
    )
    ax_cliff.errorbar(
        epsilons,
        sarsa_cliff_means,
        yerr=sarsa_cliff_stds,
        marker="o",
        linewidth=2,
        capsize=4,
        label="SARSA",
    )
    ax_cliff.set_title(f"Average Cliff Falls (Last {tail_window} Episodes)")
    ax_cliff.set_xlabel("Epsilon")
    ax_cliff.set_ylabel("Cliff Falls per Episode")
    ax_cliff.legend()
    ax_cliff.grid(alpha=0.3)

    # ---------- 5.3 Steps vs Epsilon ----------
    ax_steps.errorbar(
        epsilons,
        q_steps_means,
        yerr=q_steps_stds,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Q-learning",
    )
    ax_steps.errorbar(
        epsilons,
        sarsa_steps_means,
        yerr=sarsa_steps_stds,
        marker="o",
        linewidth=2,
        capsize=4,
        label="SARSA",
    )
    ax_steps.set_title(f"Average Episode Steps (Last {tail_window} Episodes)")
    ax_steps.set_xlabel("Epsilon")
    ax_steps.set_ylabel("Episode Steps")
    ax_steps.legend()
    ax_steps.grid(alpha=0.3)

    # ---------- 5.4 Success Rate vs Epsilon ----------
    ax_success.errorbar(
        epsilons,
        q_success_means,
        yerr=q_success_stds,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Q-learning",
    )
    ax_success.errorbar(
        epsilons,
        sarsa_success_means,
        yerr=sarsa_success_stds,
        marker="o",
        linewidth=2,
        capsize=4,
        label="SARSA",
    )
    ax_success.set_title(f"Average Success Rate (Last {tail_window} Episodes)")
    ax_success.set_xlabel("Epsilon")
    ax_success.set_ylabel("Success Rate")
    ax_success.set_ylim(0, 1.05)
    ax_success.legend()
    ax_success.grid(alpha=0.3)

    fig.suptitle(
        "Q-learning vs SARSA: Epsilon Experiment",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()
