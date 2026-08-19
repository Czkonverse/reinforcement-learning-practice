import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from envs.cliff_world import CliffWorld
from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SarsaAgent


def plot_grid(record_states, record_actions):
    state_action_dict = dict(zip(record_states, record_actions))
    print(state_action_dict)
    grid = ""
    for row in range(test_env.rows):
        for col in range(test_env.cols):
            pos_tmp = (row, col)
            if pos_tmp in state_action_dict:
                action_str = actions_dict[state_action_dict[pos_tmp]]
                grid += f" {action_str}"
            # cliff
            elif pos_tmp in test_env.cliffs:
                grid += " X"
            # start
            elif pos_tmp == test_env.start:
                grid += " S"
            # goal
            elif pos_tmp == test_env.goal:
                grid += " G"
            # passage
            else:
                grid += " ."

        # a newline
        grid += "\n"

    print(grid)


def evaluate(env, agent, max_steps):
    state = env.reset()

    total_rewards = 0
    record_actions = []
    record_states = []

    for step in range(max_steps):
        record_states.append(state)

        action = agent.select_action(state, training=False)
        next_state, reward, done = env.step(action)

        record_actions.append(int(action))
        total_rewards += reward

        if done:
            return done, step + 1, total_rewards, record_actions, record_states

        state = next_state

    return False, max_steps, total_rewards, record_actions, record_states


if __name__ == "__main__":
    actions_dict = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    num_episodes = 300
    max_steps = 200
    cliff_falls_reward = -100
    seeds = range(20)

    env = CliffWorld()
    state_shape = (env.rows, env.cols)
    num_actions = env.num_actions

    # stats
    q_episode_steps = []
    q_episode_rewards = []
    q_episode_success = []
    q_episode_cliff_falls = []

    # q learning
    q_agent = QLearningAgent(state_shape, num_actions, alpha, gamma, epsilon)

    for episode in range(num_episodes):

        state = env.reset()

        total_rewards = 0
        cliff_falls = 0
        done = False
        for step in range(max_steps):

            action = q_agent.select_action(state)

            next_state, reward, done = env.step(action)
            total_rewards += reward
            if reward == cliff_falls_reward:
                cliff_falls += 1

            q_agent.update(state, action, reward, next_state, done)

            if done:
                q_episode_steps.append(step + 1)
                break

            state = next_state

        if not done:
            q_episode_steps.append(max_steps)
        q_episode_rewards.append(total_rewards)
        q_episode_success.append(done)
        q_episode_cliff_falls.append(cliff_falls)

    # evaluate
    test_env = CliffWorld()
    state = test_env.reset()
    done, total_steps, total_rewards, record_actions, record_states = evaluate(
        test_env, q_agent, max_steps
    )
    print(
        "done, total_steps, total_rewards: \n",
        done,
        total_steps,
        total_rewards,
    )

    plot_grid(record_states, record_actions)
    print("\n" * 5)

    ########################## SARSA ##########################
    env = CliffWorld()
    state_shape = (env.rows, env.cols)
    num_actions = env.num_actions

    # stats
    sarsa_episode_steps = []
    sarsa_episode_rewards = []
    sarsa_episode_success = []
    sarsa_episode_cliff_falls = []

    sarsa_agent = SarsaAgent(state_shape, num_actions, alpha, gamma, epsilon)
    # step=0, initiate action
    for episode in range(num_episodes):
        state = env.reset()
        action = sarsa_agent.select_action(state)

        total_reward = 0
        cliff_falls = 0
        done = False
        for step in range(max_steps):

            next_state, new_reward, done = env.step(action)
            total_reward += new_reward
            if new_reward == cliff_falls_reward:
                cliff_falls += 1

            next_action = sarsa_agent.select_action(next_state)

            sarsa_agent.update(state, action, new_reward, next_state, next_action, done)

            if done:
                sarsa_episode_steps.append(step + 1)
                break

            action = next_action
            state = next_state

        if not done:
            sarsa_episode_steps.append(max_steps)

        sarsa_episode_rewards.append(total_reward)
        sarsa_episode_success.append(done)
        sarsa_episode_cliff_falls.append(cliff_falls)

    # evaluate
    test_env = CliffWorld()
    done, total_steps, total_rewards, record_actions, record_states = evaluate(
        test_env, sarsa_agent, max_steps
    )
    print(
        "done, total_steps, total_rewards: \n",
        done,
        total_steps,
        total_rewards,
    )

    plot_grid(record_states, record_actions)

    ######### train
    episodes = np.arange(1, num_episodes + 1)

    window = 20

    # 2x2 子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_reward, ax_cliff, ax_steps, ax_success = axes.ravel()

    # ---------- 1. Training Reward ----------
    ax_reward.plot(
        episodes,
        q_episode_rewards,
        color="tab:blue",
        alpha=0.2,
        linewidth=0.8,
        label="Q-learning raw",
    )
    q_smoothed = np.convolve(
        q_episode_rewards,
        np.ones(window) / window,
        mode="valid",
    )
    ax_reward.plot(
        episodes[window - 1 :],
        q_smoothed,
        color="tab:blue",
        linewidth=2.5,
        label=f"Q-learning MA{window}",
    )

    ax_reward.plot(
        episodes,
        sarsa_episode_rewards,
        color="tab:orange",
        alpha=0.2,
        linewidth=0.8,
        label="SARSA raw",
    )
    sarsa_smoothed = np.convolve(
        sarsa_episode_rewards,
        np.ones(window) / window,
        mode="valid",
    )
    ax_reward.plot(
        episodes[window - 1 :],
        sarsa_smoothed,
        color="tab:orange",
        linewidth=2.5,
        label=f"SARSA MA{window}",
    )

    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Episode Reward")
    ax_reward.set_title("Q-learning vs SARSA: Training Reward")
    ax_reward.legend()

    # ---------- 2. Cliff Falls per Episode ----------
    ax_cliff.plot(
        episodes,
        q_episode_cliff_falls,
        color="tab:blue",
        alpha=0.2,
        linewidth=0.8,
        label="Q-learning raw",
    )
    if len(q_episode_cliff_falls) >= window:
        q_cliff_ma = np.convolve(
            q_episode_cliff_falls,
            np.ones(window) / window,
            mode="valid",
        )
        ax_cliff.plot(
            episodes[window - 1 :],
            q_cliff_ma,
            color="tab:blue",
            linewidth=2.5,
            label=f"Q-learning MA{window}",
        )

    ax_cliff.plot(
        episodes,
        sarsa_episode_cliff_falls,
        color="tab:orange",
        alpha=0.2,
        linewidth=0.8,
        label="SARSA raw",
    )
    if len(sarsa_episode_cliff_falls) >= window:
        sarsa_cliff_ma = np.convolve(
            sarsa_episode_cliff_falls,
            np.ones(window) / window,
            mode="valid",
        )
        ax_cliff.plot(
            episodes[window - 1 :],
            sarsa_cliff_ma,
            color="tab:orange",
            linewidth=2.5,
            label=f"SARSA MA{window}",
        )

    ax_cliff.set_xlabel("Episode")
    ax_cliff.set_ylabel("Cliff Falls per Episode")
    ax_cliff.set_title("Q-learning vs SARSA: Cliff Falls")
    ax_cliff.legend()

    # ---------- 3. Episode Steps (Moving Average) ----------
    if len(q_episode_steps) >= window:
        q_steps_ma = np.convolve(
            q_episode_steps,
            np.ones(window) / window,
            mode="valid",
        )
        ax_steps.plot(
            episodes[window - 1 :],
            q_steps_ma,
            color="tab:blue",
            linewidth=2.5,
            label=f"Q-learning MA{window}",
        )

    if len(sarsa_episode_steps) >= window:
        sarsa_steps_ma = np.convolve(
            sarsa_episode_steps,
            np.ones(window) / window,
            mode="valid",
        )
        ax_steps.plot(
            episodes[window - 1 :],
            sarsa_steps_ma,
            color="tab:orange",
            linewidth=2.5,
            label=f"SARSA MA{window}",
        )

    ax_steps.set_xlabel("Episode")
    ax_steps.set_ylabel("Episode Steps")
    ax_steps.set_title("Q-learning vs SARSA: Episode Steps")
    ax_steps.legend()

    # ---------- 4. Rolling Success Rate ----------
    if len(q_episode_success) >= window:
        q_success_rate = np.convolve(
            np.array(q_episode_success, dtype=float),
            np.ones(window) / window,
            mode="valid",
        )
        ax_success.plot(
            episodes[window - 1 :],
            q_success_rate,
            color="tab:blue",
            linewidth=2.5,
            label=f"Q-learning rolling success rate ({window})",
        )

    if len(sarsa_episode_success) >= window:
        sarsa_success_rate = np.convolve(
            np.array(sarsa_episode_success, dtype=float),
            np.ones(window) / window,
            mode="valid",
        )
        ax_success.plot(
            episodes[window - 1 :],
            sarsa_success_rate,
            color="tab:orange",
            linewidth=2.5,
            label=f"SARSA rolling success rate ({window})",
        )

    ax_success.set_xlabel("Episode")
    ax_success.set_ylabel("Success Rate")
    ax_success.set_title("Q-learning vs SARSA: Rolling Success Rate")
    # success rate 固定在 0 ~ 1
    ax_success.set_ylim(0, 1.05)
    ax_success.legend()

    plt.tight_layout()
    plt.show()
