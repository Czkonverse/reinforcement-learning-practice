import os

import numpy as np

from envs.cliff_world import CliffWorld
from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SarsaAgent


def evaluate(
    env,
    agent,
    max_steps=200,
    cliff_falls_reward=-100,
):
    state = env.reset()

    total_reward = 0
    cliff_falls = 0

    # 初始 state 先记录
    record_states = [state]
    record_actions = []

    for step in range(max_steps):

        # greedy action
        action = agent.select_action(
            state,
            training=False,
        )

        next_state, reward, done = env.step(action)

        # 记录 transition
        record_actions.append(int(action))
        record_states.append(next_state)

        total_reward += reward

        if reward == cliff_falls_reward:
            cliff_falls += 1

        if done:
            return {
                "success": True,
                "steps": step + 1,
                "reward": total_reward,
                "cliff_falls": cliff_falls,
                "states": record_states,
                "actions": record_actions,
            }

        state = next_state

    return {
        "success": False,
        "steps": max_steps,
        "reward": total_reward,
        "cliff_falls": cliff_falls,
        "states": record_states,
        "actions": record_actions,
    }


def train_sarsa(
    env,
    seed,
    initial_epsilon,
    end_epsilon,
    decay_type="linear",
    alpha=0.1,
    gamma=0.9,
    num_episodes=300,
    max_steps=200,
    cliff_falls_reward=-100,
):
    np.random.seed(seed)

    # ------------------------------------------------
    # Stats
    # ------------------------------------------------
    sarsa_episode_steps = []
    sarsa_episode_rewards = []
    sarsa_episode_success = []
    sarsa_episode_cliff_falls = []
    sarsa_episode_epsilons = []

    # ------------------------------------------------
    # Agent
    # ------------------------------------------------
    state_shape = (env.rows, env.cols)
    num_actions = env.num_actions

    sarsa_agent = SarsaAgent(
        state_shape,
        num_actions,
        alpha,
        gamma,
        initial_epsilon,
    )

    # ------------------------------------------------
    # Training
    # ------------------------------------------------
    for episode in range(num_episodes):

        # Current epsilon
        epsilon = epsilon_decay(
            initial_epsilon=initial_epsilon,
            end_epsilon=end_epsilon,
            episode=episode,
            num_episodes=num_episodes,
            decay_type=decay_type,
        )

        sarsa_agent.epsilon = epsilon
        sarsa_episode_epsilons.append(epsilon)

        state = env.reset()

        # SARSA needs initial action
        action = sarsa_agent.select_action(state)

        total_reward = 0
        cliff_falls = 0
        done = False

        for step in range(max_steps):

            next_state, reward, done = env.step(action)

            total_reward += reward

            if reward == cliff_falls_reward:
                cliff_falls += 1

            next_action = sarsa_agent.select_action(next_state)

            sarsa_agent.update(
                state,
                action,
                reward,
                next_state,
                next_action,
                done,
            )

            if done:
                sarsa_episode_steps.append(step + 1)
                break

            state = next_state
            action = next_action

        if not done:
            sarsa_episode_steps.append(max_steps)

        sarsa_episode_rewards.append(total_reward)
        sarsa_episode_success.append(done)
        sarsa_episode_cliff_falls.append(cliff_falls)

    return (
        sarsa_episode_steps,
        sarsa_episode_rewards,
        sarsa_episode_success,
        sarsa_episode_cliff_falls,
        sarsa_episode_epsilons,
        sarsa_agent,
    )


def train_q_learning(
    env,
    seed,
    initial_epsilon,
    end_epsilon,
    decay_type="linear",
    alpha=0.1,
    gamma=0.9,
    num_episodes=300,
    max_steps=200,
    cliff_falls_reward=-100,
):
    np.random.seed(seed)

    # ------------------------------------------------
    # Stats
    # ------------------------------------------------
    q_episode_steps = []
    q_episode_rewards = []
    q_episode_success = []
    q_episode_cliff_falls = []
    q_episode_epsilons = []

    # ------------------------------------------------
    # Agent
    # ------------------------------------------------
    state_shape = (env.rows, env.cols)
    num_actions = env.num_actions

    q_agent = QLearningAgent(
        state_shape,
        num_actions,
        alpha,
        gamma,
        initial_epsilon,
    )

    # ------------------------------------------------
    # Training
    # ------------------------------------------------
    for episode in range(num_episodes):

        epsilon = epsilon_decay(
            initial_epsilon=initial_epsilon,
            end_epsilon=end_epsilon,
            episode=episode,
            num_episodes=num_episodes,
            decay_type=decay_type,
        )

        q_agent.epsilon = epsilon
        q_episode_epsilons.append(epsilon)

        state = env.reset()

        total_reward = 0
        cliff_falls = 0
        done = False

        for step in range(max_steps):

            action = q_agent.select_action(state)

            next_state, reward, done = env.step(action)

            total_reward += reward

            if reward == cliff_falls_reward:
                cliff_falls += 1

            q_agent.update(
                state,
                action,
                reward,
                next_state,
                done,
            )

            if done:
                q_episode_steps.append(step + 1)
                break

            state = next_state

        if not done:
            q_episode_steps.append(max_steps)

        q_episode_rewards.append(total_reward)
        q_episode_success.append(done)
        q_episode_cliff_falls.append(cliff_falls)

    return (
        q_episode_steps,
        q_episode_rewards,
        q_episode_success,
        q_episode_cliff_falls,
        q_episode_epsilons,
        q_agent,
    )


def epsilon_decay(
    initial_epsilon,
    end_epsilon,
    episode,
    num_episodes,
    decay_type="linear",
):
    if num_episodes < 2:
        raise ValueError("num_episodes must be >= 2 for epsilon decay.")

    if initial_epsilon < 0 or end_epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    if decay_type == "linear":

        delta = (initial_epsilon - end_epsilon) / (num_episodes - 1)

        epsilon = initial_epsilon - delta * episode

    elif decay_type == "exponential":

        if initial_epsilon == 0 or end_epsilon == 0:
            raise ValueError(
                "Exponential decay requires " "initial_epsilon > 0 and end_epsilon > 0."
            )

        decay_rate = (end_epsilon / initial_epsilon) ** (1 / (num_episodes - 1))

        epsilon = initial_epsilon * decay_rate**episode

    else:
        raise ValueError(f"Illegal decay type: {decay_type}")

    return epsilon


if __name__ == "__main__":

    # ============================================================
    # 1. Experiment config
    # ============================================================

    alpha = 0.1
    gamma = 0.9

    num_episodes = 1000
    max_steps = 200

    cliff_falls_reward = -100

    seeds = np.arange(20)

    # epsilon decay
    initial_epsilon = 0.2
    end_epsilon = 0.01
    decay_type = "linear"

    # ============================================================
    # 2. Q-learning training stats
    # ============================================================

    q_all_episodes_steps = []
    q_all_episodes_rewards = []
    q_all_episodes_success = []
    q_all_episodes_cliff_falls = []
    q_all_episodes_epsilons = []

    # Q-learning evaluation stats
    q_eval_steps = []
    q_eval_rewards = []
    q_eval_success = []
    q_eval_cliff_falls = []

    q_eval_states = []
    q_eval_actions = []

    # ============================================================
    # 3. SARSA training stats
    # ============================================================

    sarsa_all_episodes_steps = []
    sarsa_all_episodes_rewards = []
    sarsa_all_episodes_success = []
    sarsa_all_episodes_cliff_falls = []
    sarsa_all_episodes_epsilons = []

    # SARSA evaluation stats
    sarsa_eval_steps = []
    sarsa_eval_rewards = []
    sarsa_eval_success = []
    sarsa_eval_cliff_falls = []

    sarsa_eval_states = []
    sarsa_eval_actions = []

    # ============================================================
    # 4. Multi-seed experiment
    # ============================================================

    for seed in seeds:

        # ========================================================
        # 4.1 Q-learning
        # ========================================================

        q_env = CliffWorld()

        (
            q_episode_steps,
            q_episode_rewards,
            q_episode_success,
            q_episode_cliff_falls,
            q_episode_epsilons,
            q_agent,
        ) = train_q_learning(
            env=q_env,
            seed=seed,
            initial_epsilon=initial_epsilon,
            end_epsilon=end_epsilon,
            decay_type=decay_type,
            alpha=alpha,
            gamma=gamma,
            num_episodes=num_episodes,
            max_steps=max_steps,
            cliff_falls_reward=cliff_falls_reward,
        )

        q_all_episodes_steps.append(q_episode_steps)

        q_all_episodes_rewards.append(q_episode_rewards)

        q_all_episodes_success.append(q_episode_success)

        q_all_episodes_cliff_falls.append(q_episode_cliff_falls)

        q_all_episodes_epsilons.append(q_episode_epsilons)

        # --------------------------------------------------------
        # Q-learning greedy evaluation
        # --------------------------------------------------------

        q_test_env = CliffWorld()

        q_eval_result = evaluate(
            env=q_test_env,
            agent=q_agent,
            max_steps=max_steps,
            cliff_falls_reward=cliff_falls_reward,
        )

        q_eval_steps.append(q_eval_result["steps"])

        q_eval_rewards.append(q_eval_result["reward"])

        q_eval_success.append(q_eval_result["success"])

        q_eval_cliff_falls.append(q_eval_result["cliff_falls"])

        q_eval_states.append(q_eval_result["states"])

        q_eval_actions.append(q_eval_result["actions"])

        # ========================================================
        # 4.2 SARSA
        # ========================================================

        sarsa_env = CliffWorld()

        (
            sarsa_episode_steps,
            sarsa_episode_rewards,
            sarsa_episode_success,
            sarsa_episode_cliff_falls,
            sarsa_episode_epsilons,
            sarsa_agent,
        ) = train_sarsa(
            env=sarsa_env,
            seed=seed,
            initial_epsilon=initial_epsilon,
            end_epsilon=end_epsilon,
            decay_type=decay_type,
            alpha=alpha,
            gamma=gamma,
            num_episodes=num_episodes,
            max_steps=max_steps,
            cliff_falls_reward=cliff_falls_reward,
        )

        sarsa_all_episodes_steps.append(sarsa_episode_steps)

        sarsa_all_episodes_rewards.append(sarsa_episode_rewards)

        sarsa_all_episodes_success.append(sarsa_episode_success)

        sarsa_all_episodes_cliff_falls.append(sarsa_episode_cliff_falls)

        sarsa_all_episodes_epsilons.append(sarsa_episode_epsilons)

        # --------------------------------------------------------
        # SARSA greedy evaluation
        # --------------------------------------------------------

        sarsa_test_env = CliffWorld()

        sarsa_eval_result = evaluate(
            env=sarsa_test_env,
            agent=sarsa_agent,
            max_steps=max_steps,
            cliff_falls_reward=cliff_falls_reward,
        )

        sarsa_eval_steps.append(sarsa_eval_result["steps"])

        sarsa_eval_rewards.append(sarsa_eval_result["reward"])

        sarsa_eval_success.append(sarsa_eval_result["success"])

        sarsa_eval_cliff_falls.append(sarsa_eval_result["cliff_falls"])

        sarsa_eval_states.append(sarsa_eval_result["states"])

        sarsa_eval_actions.append(sarsa_eval_result["actions"])

    # ============================================================
    # 5. Convert training data to ndarray
    # ============================================================

    q_all_episodes_rewards = np.array(q_all_episodes_rewards)

    q_all_episodes_steps = np.array(q_all_episodes_steps)

    q_all_episodes_success = np.array(q_all_episodes_success)

    q_all_episodes_cliff_falls = np.array(q_all_episodes_cliff_falls)

    q_all_episodes_epsilons = np.array(q_all_episodes_epsilons)

    sarsa_all_episodes_rewards = np.array(sarsa_all_episodes_rewards)

    sarsa_all_episodes_steps = np.array(sarsa_all_episodes_steps)

    sarsa_all_episodes_success = np.array(sarsa_all_episodes_success)

    sarsa_all_episodes_cliff_falls = np.array(sarsa_all_episodes_cliff_falls)

    sarsa_all_episodes_epsilons = np.array(sarsa_all_episodes_epsilons)

    # ============================================================
    # 6. Convert evaluation statistics
    # ============================================================

    q_eval_steps = np.array(q_eval_steps)
    q_eval_rewards = np.array(q_eval_rewards)
    q_eval_success = np.array(q_eval_success)
    q_eval_cliff_falls = np.array(q_eval_cliff_falls)

    sarsa_eval_steps = np.array(sarsa_eval_steps)

    sarsa_eval_rewards = np.array(sarsa_eval_rewards)

    sarsa_eval_success = np.array(sarsa_eval_success)

    sarsa_eval_cliff_falls = np.array(sarsa_eval_cliff_falls)

    # Variable-length trajectories:
    # object arrays are required.
    q_eval_states = np.array(
        q_eval_states,
        dtype=object,
    )

    q_eval_actions = np.array(
        q_eval_actions,
        dtype=object,
    )

    sarsa_eval_states = np.array(
        sarsa_eval_states,
        dtype=object,
    )

    sarsa_eval_actions = np.array(
        sarsa_eval_actions,
        dtype=object,
    )

    # ============================================================
    # 7. Sanity checks
    # ============================================================

    print("\nTraining data shapes")

    print(
        "Q rewards:",
        q_all_episodes_rewards.shape,
    )

    print(
        "SARSA rewards:",
        sarsa_all_episodes_rewards.shape,
    )

    print(
        "Q epsilons:",
        q_all_episodes_epsilons.shape,
    )

    print(
        "SARSA epsilons:",
        sarsa_all_episodes_epsilons.shape,
    )

    # ============================================================
    # 8. Greedy evaluation summary
    # ============================================================

    print("\n" + "=" * 70)
    print("Greedy Evaluation After Linear Epsilon Decay")
    print("=" * 70)

    print("\nQ-learning")

    print(
        "steps:",
        q_eval_steps,
    )

    print(f"success rate: " f"{q_eval_success.mean():.3f}")

    print(f"steps: " f"{q_eval_steps.mean():.2f} " f"± {q_eval_steps.std(ddof=1):.2f}")

    print(
        f"reward: "
        f"{q_eval_rewards.mean():.2f} "
        f"± {q_eval_rewards.std(ddof=1):.2f}"
    )

    print(
        f"cliff falls: "
        f"{q_eval_cliff_falls.mean():.3f} "
        f"± {q_eval_cliff_falls.std(ddof=1):.3f}"
    )

    print("\nSARSA")

    print(
        "steps:",
        sarsa_eval_steps,
    )

    print(f"success rate: " f"{sarsa_eval_success.mean():.3f}")

    print(
        f"steps: "
        f"{sarsa_eval_steps.mean():.2f} "
        f"± {sarsa_eval_steps.std(ddof=1):.2f}"
    )

    print(
        f"reward: "
        f"{sarsa_eval_rewards.mean():.2f} "
        f"± {sarsa_eval_rewards.std(ddof=1):.2f}"
    )

    print(
        f"cliff falls: "
        f"{sarsa_eval_cliff_falls.mean():.3f} "
        f"± {sarsa_eval_cliff_falls.std(ddof=1):.3f}"
    )

    # ============================================================
    # 9. Save
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
        "q_learning_vs_sarsa_linear_decay.npz",
    )

    np.savez_compressed(
        save_path,
        # --------------------------------------------------------
        # Experiment config
        # --------------------------------------------------------
        seeds=seeds,
        alpha=alpha,
        gamma=gamma,
        num_episodes=num_episodes,
        max_steps=max_steps,
        cliff_falls_reward=cliff_falls_reward,
        initial_epsilon=initial_epsilon,
        end_epsilon=end_epsilon,
        decay_type=decay_type,
        # --------------------------------------------------------
        # Q-learning training
        # --------------------------------------------------------
        q_rewards=q_all_episodes_rewards,
        q_steps=q_all_episodes_steps,
        q_success=q_all_episodes_success,
        q_cliff_falls=q_all_episodes_cliff_falls,
        q_epsilons=q_all_episodes_epsilons,
        # Q-learning evaluation
        q_eval_steps=q_eval_steps,
        q_eval_rewards=q_eval_rewards,
        q_eval_success=q_eval_success,
        q_eval_cliff_falls=q_eval_cliff_falls,
        q_eval_states=q_eval_states,
        q_eval_actions=q_eval_actions,
        # --------------------------------------------------------
        # SARSA training
        # --------------------------------------------------------
        sarsa_rewards=sarsa_all_episodes_rewards,
        sarsa_steps=sarsa_all_episodes_steps,
        sarsa_success=sarsa_all_episodes_success,
        sarsa_cliff_falls=sarsa_all_episodes_cliff_falls,
        sarsa_epsilons=sarsa_all_episodes_epsilons,
        # SARSA evaluation
        sarsa_eval_steps=sarsa_eval_steps,
        sarsa_eval_rewards=sarsa_eval_rewards,
        sarsa_eval_success=sarsa_eval_success,
        sarsa_eval_cliff_falls=sarsa_eval_cliff_falls,
        sarsa_eval_states=sarsa_eval_states,
        sarsa_eval_actions=sarsa_eval_actions,
    )

    print(f"\nSaved experiment data to:\n{save_path}")
