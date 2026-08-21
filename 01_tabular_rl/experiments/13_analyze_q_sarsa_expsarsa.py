import os

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Config
# ============================================================

WINDOW = 20
LAST_N_EPISODES = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, "resource")

DATA_PATH = os.path.join(
    RESOURCE_DIR,
    "q_sarsa_expsarsa_eps-0.2_seeds-20.npz",
)

FIGURE_DIR = os.path.join(RESOURCE_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================


def moving_average(values, window):
    """
    Compute moving average for a 1D array.

    Example:
        values = [1, 2, 3, 4, 5]
        window = 3

        result:
        [2, 3, 4]
    """
    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def rolling_mean_2d(values, window):
    """
    Compute rolling mean independently for every seed.

    Input shape:
        (num_seeds, num_episodes)

    Output shape:
        (num_seeds, num_episodes - window + 1)
    """
    return np.array([moving_average(seed_values, window) for seed_values in values])


def mean_and_standard_error(values):
    """
    Calculate mean and standard error across seeds.

    Input:
        values.shape = (num_seeds, num_points)

    Returns:
        mean.shape = (num_points,)
        se.shape   = (num_points,)
    """
    mean = np.mean(values, axis=0)

    std = np.std(
        values,
        axis=0,
        ddof=1,
    )

    standard_error = std / np.sqrt(values.shape[0])

    return mean, standard_error


def plot_metric(
    algorithms,
    ylabel,
    title,
    filename,
    window=20,
):
    """
    Plot:
        moving average over episodes
        +
        mean across seeds
        +
        standard-error uncertainty band

    algorithms:
        {
            "Q-learning": array,
            "SARSA": array,
            "Expected SARSA": array,
        }

    Each array shape:
        (num_seeds, num_episodes)
    """

    plt.figure(figsize=(10, 6))

    for label, values in algorithms.items():

        # First smooth each individual seed.
        rolling_values = rolling_mean_2d(
            values,
            window,
        )

        # Then compute cross-seed mean and standard error.
        mean, se = mean_and_standard_error(rolling_values)

        episodes = np.arange(
            window,
            values.shape[1] + 1,
        )

        # Mean line
        line = plt.plot(
            episodes,
            mean,
            label=label,
            linewidth=2,
        )[0]

        # Standard-error band
        plt.fill_between(
            episodes,
            mean - se,
            mean + se,
            alpha=0.2,
            color=line.get_color(),
        )

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(
        FIGURE_DIR,
        filename,
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(f"Saved figure: {save_path}")


def rolling_success_rate(success_values, window):
    """
    success_values:
        bool array
        shape = (num_seeds, num_episodes)

    For each seed:
        rolling success rate =
        success count in last `window` episodes / window

    Since bool:
        True  -> 1
        False -> 0

    therefore rolling mean = rolling success rate.
    """

    return rolling_mean_2d(
        success_values.astype(float),
        window,
    )


def plot_success_rate(
    algorithms,
    window=20,
):
    plt.figure(figsize=(10, 6))

    for label, values in algorithms.items():

        rolling_values = rolling_success_rate(
            values,
            window,
        )

        mean, se = mean_and_standard_error(rolling_values)

        episodes = np.arange(
            window,
            values.shape[1] + 1,
        )

        line = plt.plot(
            episodes,
            mean,
            linewidth=2,
            label=label,
        )[0]

        plt.fill_between(
            episodes,
            mean - se,
            mean + se,
            alpha=0.2,
            color=line.get_color(),
        )

    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title(f"Rolling Success Rate " f"(window={window}, mean across seeds)")

    plt.ylim(-0.02, 1.02)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(
        FIGURE_DIR,
        "success_rate.png",
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(f"Saved figure: {save_path}")


# ============================================================
# Final statistics
# ============================================================


def summarize_last_episodes(
    rewards,
    steps,
    success,
    cliff_falls,
    last_n=100,
):
    """
    First calculate each seed's average over the last N episodes,
    then calculate mean/std across seeds.

    This keeps the seed as the independent experimental unit.
    """

    reward_per_seed = np.mean(
        rewards[:, -last_n:],
        axis=1,
    )

    steps_per_seed = np.mean(
        steps[:, -last_n:],
        axis=1,
    )

    success_per_seed = np.mean(
        success[:, -last_n:],
        axis=1,
    )

    cliff_per_seed = np.mean(
        cliff_falls[:, -last_n:],
        axis=1,
    )

    return {
        "reward_mean": np.mean(reward_per_seed),
        "reward_std": np.std(
            reward_per_seed,
            ddof=1,
        ),
        "steps_mean": np.mean(steps_per_seed),
        "steps_std": np.std(
            steps_per_seed,
            ddof=1,
        ),
        "success_mean": np.mean(success_per_seed),
        "success_std": np.std(
            success_per_seed,
            ddof=1,
        ),
        "cliff_mean": np.mean(cliff_per_seed),
        "cliff_std": np.std(
            cliff_per_seed,
            ddof=1,
        ),
    }


def print_summary_table(results):
    print()
    print("=" * 100)
    print(f"Final performance " f"(last {LAST_N_EPISODES} episodes)")
    print("=" * 100)

    header = (
        f"{'Algorithm':<20}"
        f"{'Reward':>20}"
        f"{'Steps':>20}"
        f"{'Success Rate':>20}"
        f"{'Cliff Falls':>20}"
    )

    print(header)
    print("-" * 100)

    for algorithm, stats in results.items():

        reward = f"{stats['reward_mean']:.2f} " f"± {stats['reward_std']:.2f}"

        steps = f"{stats['steps_mean']:.2f} " f"± {stats['steps_std']:.2f}"

        success = f"{stats['success_mean']:.3f} " f"± {stats['success_std']:.3f}"

        cliff = f"{stats['cliff_mean']:.3f} " f"± {stats['cliff_std']:.3f}"

        print(
            f"{algorithm:<20}"
            f"{reward:>20}"
            f"{steps:>20}"
            f"{success:>20}"
            f"{cliff:>20}"
        )

    print("=" * 100)
    print("Values are mean ± std across seeds.")


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    data = np.load(DATA_PATH)

    print("Loaded:")
    print(DATA_PATH)

    print()
    print("Keys:")
    print(data.files)

    # --------------------------------------------------------
    # 2. Experiment config
    # --------------------------------------------------------

    seeds = data["seeds"]
    alpha = float(data["alpha"])
    gamma = float(data["gamma"])
    epsilon = float(data["epsilon"])
    num_episodes = int(data["num_episodes"])
    max_steps = int(data["max_steps"])
    cliff_falls_reward = int(data["cliff_falls_reward"])

    print()
    print("Experiment config")
    print("-----------------")
    print(f"num seeds:          {len(seeds)}")
    print(f"num episodes:       {num_episodes}")
    print(f"alpha:              {alpha}")
    print(f"gamma:              {gamma}")
    print(f"epsilon:            {epsilon}")
    print(f"max steps:          {max_steps}")
    print(f"cliff fall reward:  {cliff_falls_reward}")

    # --------------------------------------------------------
    # 3. Q-learning
    # --------------------------------------------------------

    q_rewards = data["q_rewards"]
    q_steps = data["q_steps"]
    q_success = data["q_success"]
    q_cliff_falls = data["q_cliff_falls"]

    # --------------------------------------------------------
    # 4. SARSA
    # --------------------------------------------------------

    sarsa_rewards = data["sarsa_rewards"]
    sarsa_steps = data["sarsa_steps"]
    sarsa_success = data["sarsa_success"]
    sarsa_cliff_falls = data["sarsa_cliff_falls"]

    # --------------------------------------------------------
    # 5. Expected SARSA
    # --------------------------------------------------------

    exp_sarsa_rewards = data["exp_sarsa_rewards"]
    exp_sarsa_steps = data["exp_sarsa_steps"]
    exp_sarsa_success = data["exp_sarsa_success"]
    exp_sarsa_cliff_falls = data["exp_sarsa_cliff_falls"]

    # --------------------------------------------------------
    # 6. Validate shapes
    # --------------------------------------------------------

    expected_shape = (
        len(seeds),
        num_episodes,
    )

    arrays = {
        "q_rewards": q_rewards,
        "q_steps": q_steps,
        "q_success": q_success,
        "q_cliff_falls": q_cliff_falls,
        "sarsa_rewards": sarsa_rewards,
        "sarsa_steps": sarsa_steps,
        "sarsa_success": sarsa_success,
        "sarsa_cliff_falls": sarsa_cliff_falls,
        "exp_sarsa_rewards": exp_sarsa_rewards,
        "exp_sarsa_steps": exp_sarsa_steps,
        "exp_sarsa_success": exp_sarsa_success,
        "exp_sarsa_cliff_falls": exp_sarsa_cliff_falls,
    }

    print()
    print("Array shapes")
    print("------------")

    for name, array in arrays.items():
        print(f"{name:<25} " f"{array.shape}")

        assert array.shape == expected_shape, (
            f"{name} has shape {array.shape}, " f"expected {expected_shape}"
        )

    print()
    print("All shapes are correct.")

    # --------------------------------------------------------
    # 7. Reward
    # --------------------------------------------------------

    reward_data = {
        "Q-learning": q_rewards,
        "SARSA": sarsa_rewards,
        "Expected SARSA": exp_sarsa_rewards,
    }

    plot_metric(
        reward_data,
        ylabel="Episode Reward",
        title=(f"Episode Reward " f"(ε={epsilon}, {len(seeds)} seeds)"),
        filename="reward.png",
        window=WINDOW,
    )

    # --------------------------------------------------------
    # 8. Cliff falls
    # --------------------------------------------------------

    cliff_data = {
        "Q-learning": q_cliff_falls,
        "SARSA": sarsa_cliff_falls,
        "Expected SARSA": exp_sarsa_cliff_falls,
    }

    plot_metric(
        cliff_data,
        ylabel="Cliff Falls per Episode",
        title=(f"Cliff Falls " f"(ε={epsilon}, {len(seeds)} seeds)"),
        filename="cliff_falls.png",
        window=WINDOW,
    )

    # --------------------------------------------------------
    # 9. Episode steps
    # --------------------------------------------------------

    steps_data = {
        "Q-learning": q_steps,
        "SARSA": sarsa_steps,
        "Expected SARSA": exp_sarsa_steps,
    }

    plot_metric(
        steps_data,
        ylabel="Steps per Episode",
        title=(f"Episode Steps " f"(ε={epsilon}, {len(seeds)} seeds)"),
        filename="steps.png",
        window=WINDOW,
    )

    # --------------------------------------------------------
    # 10. Rolling success rate
    # --------------------------------------------------------

    success_data = {
        "Q-learning": q_success,
        "SARSA": sarsa_success,
        "Expected SARSA": exp_sarsa_success,
    }

    plot_success_rate(
        success_data,
        window=WINDOW,
    )

    # --------------------------------------------------------
    # 11. Final performance
    # --------------------------------------------------------

    summary = {
        "Q-learning": summarize_last_episodes(
            q_rewards,
            q_steps,
            q_success,
            q_cliff_falls,
            last_n=LAST_N_EPISODES,
        ),
        "SARSA": summarize_last_episodes(
            sarsa_rewards,
            sarsa_steps,
            sarsa_success,
            sarsa_cliff_falls,
            last_n=LAST_N_EPISODES,
        ),
        "Expected SARSA": summarize_last_episodes(
            exp_sarsa_rewards,
            exp_sarsa_steps,
            exp_sarsa_success,
            exp_sarsa_cliff_falls,
            last_n=LAST_N_EPISODES,
        ),
    }

    print_summary_table(summary)
