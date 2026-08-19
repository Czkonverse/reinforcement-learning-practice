import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_cliff_world(ax, rows=4, cols=12):
    """
    Draw base CliffWorld grid.

    Assumed layout:
    - start = (3, 0)
    - goal  = (3, 11)
    - cliff = (3, 1) ~ (3, 10)
    """
    start = (3, 0)
    goal = (3, cols - 1)
    cliff_cells = [(3, c) for c in range(1, cols - 1)]

    # background cells
    for r in range(rows):
        for c in range(cols):
            y = rows - 1 - r

            # cliff cells
            if (r, c) in cliff_cells:
                rect = Rectangle(
                    (c, y),
                    1,
                    1,
                    alpha=0.3,
                )
                ax.add_patch(rect)

            # start cell
            if (r, c) == start:
                ax.text(
                    c + 0.5,
                    y + 0.5,
                    "S",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )

            # goal cell
            if (r, c) == goal:
                ax.text(
                    c + 0.5,
                    y + 0.5,
                    "G",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )

    # grid lines
    for x in range(cols + 1):
        ax.plot([x, x], [0, rows], linewidth=1)

    for y in range(rows + 1):
        ax.plot([0, cols], [y, y], linewidth=1)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels(np.arange(cols))

    # row label: top row is 0
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(np.arange(rows - 1, -1, -1))

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")


def states_to_xy(states, rows):
    """
    Convert grid states [(r, c), ...] to plot coordinates.
    """
    xs = []
    ys = []

    for r, c in states:
        x = c + 0.5
        y = rows - r - 0.5
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def plot_trajectory(ax, states, title, rows=4, cols=12):
    """
    Draw one trajectory on CliffWorld.
    """
    draw_cliff_world(ax, rows=rows, cols=cols)

    xs, ys = states_to_xy(states, rows)

    ax.plot(
        xs,
        ys,
        marker="o",
        linewidth=2,
        markersize=5,
    )

    # annotate step index on each state
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(
            x,
            y + 0.12,
            str(i),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(title)


if __name__ == "__main__":
    # ------------------------------------------------------------
    # 1. Load saved experiment data
    # ------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        base_dir,
        "resource",
        "q_learning_vs_sarsa_linear_decay.npz",
    )

    data = np.load(data_path, allow_pickle=True)

    # ------------------------------------------------------------
    # 2. Extract greedy evaluation trajectories
    # ------------------------------------------------------------
    q_eval_states = data["q_eval_states"]
    q_eval_steps = data["q_eval_steps"]
    q_eval_success = data["q_eval_success"]

    sarsa_eval_states = data["sarsa_eval_states"]
    sarsa_eval_steps = data["sarsa_eval_steps"]
    sarsa_eval_success = data["sarsa_eval_success"]

    # Choose the first successful trajectory from each algorithm
    q_idx = np.where(q_eval_success)[0][0]
    sarsa_idx = np.where(sarsa_eval_success)[0][0]

    q_states = q_eval_states[q_idx]
    sarsa_states = sarsa_eval_states[sarsa_idx]

    q_steps = q_eval_steps[q_idx]
    sarsa_steps = sarsa_eval_steps[sarsa_idx]

    # ------------------------------------------------------------
    # 3. Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_trajectory(
        axes[0],
        q_states,
        title=f"Q-learning Greedy Trajectory ({q_steps} steps)",
        rows=4,
        cols=12,
    )

    plot_trajectory(
        axes[1],
        sarsa_states,
        title=f"SARSA Greedy Trajectory ({sarsa_steps} steps)",
        rows=4,
        cols=12,
    )

    fig.suptitle(
        "Greedy Policies After Linear Epsilon Decay",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()
