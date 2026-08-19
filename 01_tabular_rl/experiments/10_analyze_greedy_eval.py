import os
import numpy as np

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        base_dir,
        "resource",
        "q_learning_vs_sarsa_linear_decay.npz",
    )

    data = np.load(
        data_path,
        allow_pickle=True,
    )

    seeds = data["seeds"]

    q_steps = data["q_eval_steps"]
    q_success = data["q_eval_success"]
    q_states = data["q_eval_states"]
    q_actions = data["q_eval_actions"]

    sarsa_steps = data["sarsa_eval_steps"]
    sarsa_success = data["sarsa_eval_success"]
    sarsa_states = data["sarsa_eval_states"]
    sarsa_actions = data["sarsa_eval_actions"]

    # ============================================================
    # 1. Q-learning failed seeds
    # ============================================================

    print("=" * 70)
    print("Q-learning failed greedy evaluations")
    print("=" * 70)

    q_failed_indices = np.where(~q_success)[0]

    for i in q_failed_indices:

        print(f"\nSeed: {seeds[i]}")
        print(f"Steps: {q_steps[i]}")

        states = q_states[i]
        actions = q_actions[i]

        print("First 30 states:")
        print(states[:30])

        print("First 30 actions:")
        print(actions[:30])

    # ============================================================
    # 2. SARSA failed seeds
    # ============================================================

    print("\n" + "=" * 70)
    print("SARSA failed greedy evaluations")
    print("=" * 70)

    sarsa_failed_indices = np.where(~sarsa_success)[0]

    for i in sarsa_failed_indices:

        print(f"\nSeed: {seeds[i]}")
        print(f"Steps: {sarsa_steps[i]}")

        states = sarsa_states[i]
        actions = sarsa_actions[i]

        print("First 30 states:")
        print(states[:30])

        print("First 30 actions:")
        print(actions[:30])

    # ============================================================
    # 3. Successful path lengths only
    # ============================================================

    q_success_steps = q_steps[q_success]
    sarsa_success_steps = sarsa_steps[sarsa_success]

    print("\n" + "=" * 70)
    print("Successful Greedy Policies Only")
    print("=" * 70)

    print("\nQ-learning")
    print("steps:", q_success_steps)
    print(f"mean: {q_success_steps.mean():.2f}")
    print(f"min/max: " f"{q_success_steps.min()} / " f"{q_success_steps.max()}")

    print("\nSARSA")
    print("steps:", sarsa_success_steps)
    print(f"mean: {sarsa_success_steps.mean():.2f}")
    print(f"min/max: " f"{sarsa_success_steps.min()} / " f"{sarsa_success_steps.max()}")
