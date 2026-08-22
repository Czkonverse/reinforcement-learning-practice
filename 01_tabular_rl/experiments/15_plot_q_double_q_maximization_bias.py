import os

import matplotlib.pyplot as plt
import numpy as np


def load_experiment_data(data_path):
    data = np.load(data_path)

    # ============================================================
    # Experiment config
    # ============================================================
    alpha = float(data["alpha"])
    gamma = float(data["gamma"])
    epsilon = float(data["epsilon"])
    num_episodes = int(data["num_episodes"])

    reward_mean = float(data["reward_mean"])
    reward_std = float(data["reward_std"])

    seeds = data["seeds"]

    # ============================================================
    # Q-learning
    # ============================================================
    q_entered_b_records = data["q_entered_b_records"]
    q_entered_b_rate = data["q_entered_b_rate"]

    q_a1_records = data["q_a1_records"]
    q_a1_mean = data["q_a1_mean"]

    # ============================================================
    # Double Q-learning
    # ============================================================
    double_q_entered_b_records = data["double_q_entered_b_records"]
    double_q_entered_b_rate = data["double_q_entered_b_rate"]

    double_q_a1_records = data["double_q_a1_records"]
    double_q_a1_mean = data["double_q_a1_mean"]

    # ============================================================
    # Reference values
    # ============================================================
    if "true_q_a1" in data:
        true_q_a1 = float(data["true_q_a1"])
    else:
        # gamma = 1.0 in this experiment,
        # so Q*(A, action 1) = reward_mean = -0.1
        true_q_a1 = reward_mean

    if "epsilon_exploration_baseline" in data:
        epsilon_baseline = float(data["epsilon_exploration_baseline"])
    else:
        epsilon_baseline = epsilon / 2.0

    return {
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "num_episodes": num_episodes,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "seeds": seeds,
        "q_entered_b_records": q_entered_b_records,
        "q_entered_b_rate": q_entered_b_rate,
        "q_a1_records": q_a1_records,
        "q_a1_mean": q_a1_mean,
        "double_q_entered_b_records": (double_q_entered_b_records),
        "double_q_entered_b_rate": (double_q_entered_b_rate),
        "double_q_a1_records": double_q_a1_records,
        "double_q_a1_mean": double_q_a1_mean,
        "true_q_a1": true_q_a1,
        "epsilon_baseline": epsilon_baseline,
    }


def plot_entered_b_rate(
    episodes,
    q_entered_b_rate,
    double_q_entered_b_rate,
    epsilon_baseline,
    save_path,
):
    """
    Figure 1:
    Probability of entering B.

    This visualizes the behavioral consequence
    of maximization bias.
    """

    plt.figure(figsize=(12, 7))

    plt.plot(
        episodes,
        q_entered_b_rate,
        label="Q-learning",
        linewidth=2,
    )

    plt.plot(
        episodes,
        double_q_entered_b_rate,
        label="Double Q-learning",
        linewidth=2,
    )

    plt.axhline(
        y=epsilon_baseline,
        linestyle="--",
        linewidth=1.5,
        label=("Epsilon exploration baseline " f"({epsilon_baseline:.2f})"),
    )

    plt.xlabel(
        "Episode",
        fontsize=12,
    )

    plt.ylabel(
        "Probability of Entering B",
        fontsize=12,
    )

    plt.title(
        "Maximization Bias: " "Q-learning vs Double Q-learning",
        fontsize=14,
    )

    plt.xlim(
        1,
        len(episodes),
    )

    plt.ylim(
        0,
        1.0,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def plot_q_a1_estimate(
    episodes,
    q_a1_mean,
    double_q_a1_mean,
    true_q_a1,
    save_path,
):
    """
    Figure 2:
    Estimated Q(A, action 1).

    True value:
        Q*(A, action 1) = -0.1

    This directly visualizes overestimation bias.
    """

    plt.figure(figsize=(12, 7))

    plt.plot(
        episodes,
        q_a1_mean,
        label="Q-learning",
        linewidth=2,
    )

    plt.plot(
        episodes,
        double_q_a1_mean,
        label="Double Q-learning",
        linewidth=2,
    )

    # True Q(A, action 1)
    plt.axhline(
        y=true_q_a1,
        linestyle="--",
        linewidth=1.5,
        label=f"True Q(A, action 1) = {true_q_a1:.2f}",
    )

    # Zero reference line
    plt.axhline(
        y=0.0,
        linestyle=":",
        linewidth=1.0,
        label="Zero",
    )

    plt.xlabel(
        "Episode",
        fontsize=12,
    )

    plt.ylabel(
        "Estimated Q(A, action 1)",
        fontsize=12,
    )

    plt.title(
        "Overestimation of Q(A, action 1): " "Q-learning vs Double Q-learning",
        fontsize=14,
    )

    plt.xlim(
        1,
        len(episodes),
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def main():
    # ============================================================
    # Paths
    # ============================================================
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        current_dir,
        "resource",
        "q_double_q_max_bias_eps-0.1_seeds-200.npz",
    )

    figure_dir = os.path.join(
        current_dir,
        "figures",
    )

    os.makedirs(
        figure_dir,
        exist_ok=True,
    )

    # ============================================================
    # Load data
    # ============================================================
    data = load_experiment_data(data_path)

    num_episodes = data["num_episodes"]

    episodes = np.arange(
        1,
        num_episodes + 1,
    )

    # ============================================================
    # Print experiment config
    # ============================================================
    print("=" * 80)
    print("Experiment configuration")
    print("=" * 80)

    print(f"Number of seeds : {len(data['seeds'])}")
    print(f"Number episodes : {num_episodes}")
    print(f"alpha           : {data['alpha']}")
    print(f"gamma           : {data['gamma']}")
    print(f"epsilon         : {data['epsilon']}")
    print(f"reward mean     : {data['reward_mean']}")
    print(f"reward std      : {data['reward_std']}")
    print(f"true Q(A, 1)    : {data['true_q_a1']}")

    # ============================================================
    # Figure 1
    # Entered-B rate
    # ============================================================
    entered_b_figure_path = os.path.join(
        figure_dir,
        "q_double_q_entered_b_rate.png",
    )

    plot_entered_b_rate(
        episodes=episodes,
        q_entered_b_rate=data["q_entered_b_rate"],
        double_q_entered_b_rate=data["double_q_entered_b_rate"],
        epsilon_baseline=data["epsilon_baseline"],
        save_path=entered_b_figure_path,
    )

    # ============================================================
    # Figure 2
    # Estimated Q(A, action 1)
    # ============================================================
    q_estimate_figure_path = os.path.join(
        figure_dir,
        "q_double_q_a1_estimate.png",
    )

    plot_q_a1_estimate(
        episodes=episodes,
        q_a1_mean=data["q_a1_mean"],
        double_q_a1_mean=data["double_q_a1_mean"],
        true_q_a1=data["true_q_a1"],
        save_path=q_estimate_figure_path,
    )

    print()
    print("=" * 80)
    print("Figures saved")
    print("=" * 80)

    print(entered_b_figure_path)

    print(q_estimate_figure_path)


if __name__ == "__main__":
    main()
