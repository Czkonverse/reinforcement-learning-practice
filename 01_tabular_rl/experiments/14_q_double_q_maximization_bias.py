import os
import numpy as np

from envs.maximization_bias import MaximizationBiasEnv
from agents.q_learning_agent_v2 import QLearningAgentV2
from agents.double_q_learning_agent import DoubleQLearning


def train(
    agent_type,
    seeds,
    num_episodes=300,
    alpha=0.1,
    gamma=1.0,
    epsilon=0.1,
    reward_mean=-0.1,
    reward_std=1.0,
):
    num_seeds = len(seeds)

    # ============================================================
    # Record 1:
    # Whether the agent enters B in each episode
    #
    # shape = (num_seeds, num_episodes)
    #
    # 0: A -> action 0 -> Terminal
    # 1: A -> action 1 -> B
    # ============================================================
    entered_b_records = np.zeros(
        (num_seeds, num_episodes),
        dtype=int,
    )

    # ============================================================
    # Record 2:
    # Estimated Q(A, action 1) after each episode
    #
    # True value:
    # Q*(A, action 1) = -0.1
    #
    # shape = (num_seeds, num_episodes)
    # ============================================================
    q_a1_records = np.zeros(
        (num_seeds, num_episodes),
        dtype=float,
    )

    # ============================================================
    # Multiple random seeds
    # ============================================================
    for seed_idx, seed in enumerate(seeds):

        # --------------------------------------------------------
        # Environment
        # --------------------------------------------------------
        env = MaximizationBiasEnv(
            reward_mean=reward_mean,
            reward_std=reward_std,
            seed=int(seed),
        )

        # --------------------------------------------------------
        # Agent
        # --------------------------------------------------------
        if agent_type == "q_learning":

            agent = QLearningAgentV2(
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                seed=int(seed) + 10000,
            )

        elif agent_type == "double_q_learning":

            agent = DoubleQLearning(
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                seed=int(seed) + 10000,
            )

        else:
            raise ValueError(f"Invalid agent_type: {agent_type}")

        # ========================================================
        # Episodes
        # ========================================================
        for episode in range(num_episodes):

            state = env.reset()
            done = False

            while not done:

                # epsilon-greedy during training
                action = agent.select_action(
                    state,
                    training=True,
                )

                # ------------------------------------------------
                # Record whether the agent enters B
                # ------------------------------------------------
                if state == env.A:
                    entered_b_records[seed_idx, episode] = 1 if action == 1 else 0

                # ------------------------------------------------
                # Environment transition
                # ------------------------------------------------
                next_state, reward, done = env.step(action)

                # ------------------------------------------------
                # Agent update
                # ------------------------------------------------
                agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                state = next_state

            # ====================================================
            # Record estimated Q(A, action 1)
            # AFTER this episode
            # ====================================================
            if agent_type == "q_learning":

                q_a1_records[seed_idx, episode] = agent.q_table[env.A][1]

            elif agent_type == "double_q_learning":

                # Q1 and Q2 both estimate Q(A, 1),
                # so use their average as the value estimate.
                q_a1_records[seed_idx, episode] = (
                    agent.q_table_1[env.A][1] + agent.q_table_2[env.A][1]
                ) / 2.0

    return entered_b_records, q_a1_records


if __name__ == "__main__":

    # ============================================================
    # Experiment configuration
    # ============================================================
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1

    num_seeds = 200
    num_episodes = 300

    reward_mean = -0.1
    reward_std = 1.0

    seeds = np.arange(num_seeds)

    # ============================================================
    # Q-learning
    # ============================================================
    (
        q_entered_b_records,
        q_a1_records,
    ) = train(
        agent_type="q_learning",
        seeds=seeds,
        num_episodes=num_episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        reward_mean=reward_mean,
        reward_std=reward_std,
    )

    # Entered-B rate for each episode
    q_entered_b_rate = np.mean(
        q_entered_b_records,
        axis=0,
    )

    # Mean estimated Q(A, action 1) for each episode
    q_a1_mean = np.mean(
        q_a1_records,
        axis=0,
    )

    # ============================================================
    # Double Q-learning
    # ============================================================
    (
        double_q_entered_b_records,
        double_q_a1_records,
    ) = train(
        agent_type="double_q_learning",
        seeds=seeds,
        num_episodes=num_episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        reward_mean=reward_mean,
        reward_std=reward_std,
    )

    # Entered-B rate for each episode
    double_q_entered_b_rate = np.mean(
        double_q_entered_b_records,
        axis=0,
    )

    # Mean estimated Q(A, action 1) for each episode
    double_q_a1_mean = np.mean(
        double_q_a1_records,
        axis=0,
    )

    # ============================================================
    # Print summary
    # ============================================================
    print("=" * 80)
    print("Q-learning")
    print("=" * 80)

    print("entered_b_records shape:")
    print(q_entered_b_records.shape)

    print("\nFirst 20 entered-B rates:")
    print(q_entered_b_rate[:20])

    print("\nLast 20 entered-B rates:")
    print(q_entered_b_rate[-20:])

    print(
        "\nAverage entered-B rate:",
        np.mean(q_entered_b_rate),
    )

    print("\nFirst 20 mean Q(A, 1):")
    print(q_a1_mean[:20])

    print("\nLast 20 mean Q(A, 1):")
    print(q_a1_mean[-20:])

    print(
        "\nFinal mean Q(A, 1):",
        q_a1_mean[-1],
    )

    print()

    print("=" * 80)
    print("Double Q-learning")
    print("=" * 80)

    print("entered_b_records shape:")
    print(double_q_entered_b_records.shape)

    print("\nFirst 20 entered-B rates:")
    print(double_q_entered_b_rate[:20])

    print("\nLast 20 entered-B rates:")
    print(double_q_entered_b_rate[-20:])

    print(
        "\nAverage entered-B rate:",
        np.mean(double_q_entered_b_rate),
    )

    print("\nFirst 20 mean Q(A, 1):")
    print(double_q_a1_mean[:20])

    print("\nLast 20 mean Q(A, 1):")
    print(double_q_a1_mean[-20:])

    print(
        "\nFinal mean Q(A, 1):",
        double_q_a1_mean[-1],
    )

    # ============================================================
    # Save experiment
    # ============================================================
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "resource",
    )

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    save_path = os.path.join(
        save_dir,
        (f"q_double_q_max_bias_" f"eps-{epsilon}_" f"seeds-{num_seeds}.npz"),
    )

    np.savez_compressed(
        save_path,
        # --------------------------------------------------------
        # Experiment configuration
        # --------------------------------------------------------
        seeds=seeds,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        num_episodes=num_episodes,
        reward_mean=reward_mean,
        reward_std=reward_std,
        # True values / references
        true_q_a1=reward_mean,
        epsilon_exploration_baseline=epsilon / 2.0,
        # --------------------------------------------------------
        # Q-learning
        # --------------------------------------------------------
        q_entered_b_records=q_entered_b_records,
        q_entered_b_rate=q_entered_b_rate,
        q_a1_records=q_a1_records,
        q_a1_mean=q_a1_mean,
        # --------------------------------------------------------
        # Double Q-learning
        # --------------------------------------------------------
        double_q_entered_b_records=double_q_entered_b_records,
        double_q_entered_b_rate=double_q_entered_b_rate,
        double_q_a1_records=double_q_a1_records,
        double_q_a1_mean=double_q_a1_mean,
    )

    print()
    print("=" * 80)
    print(f"Experiment data saved to:\n{save_path}")
    print("=" * 80)
