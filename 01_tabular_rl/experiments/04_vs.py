import os

import numpy as np
from envs.cliff_world import CliffWorld
from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SarsaAgent

if __name__ == "__main__":

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    num_episodes = 300
    max_steps = 200
    cliff_falls_reward = -100

    # stats
    q_all_episodes_steps = []
    q_all_episodes_rewards = []
    q_all_episodes_success = []
    q_all_episodes_cliff_falls = []

    seeds = np.arange(20)

    for seed in seeds:
        np.random.seed(seed)

        # stats
        q_episode_steps = []
        q_episode_rewards = []
        q_episode_success = []
        q_episode_cliff_falls = []

        # q learning
        env = CliffWorld()
        state_shape = (env.rows, env.cols)
        num_actions = env.num_actions

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

        q_all_episodes_steps.append(q_episode_steps)
        q_all_episodes_rewards.append(q_episode_rewards)
        q_all_episodes_success.append(q_episode_success)
        q_all_episodes_cliff_falls.append(q_episode_cliff_falls)

    q_all_episodes_rewards = np.array(q_all_episodes_rewards)
    q_all_episodes_steps = np.array(q_all_episodes_steps)
    q_all_episodes_success = np.array(q_all_episodes_success)
    q_all_episodes_cliff_falls = np.array(q_all_episodes_cliff_falls)
    ########################## SARSA ##########################

    sarsa_all_episodes_steps = []
    sarsa_all_episodes_rewards = []
    sarsa_all_episodes_success = []
    sarsa_all_episodes_cliff_falls = []

    seeds = np.arange(20)

    for seed in seeds:
        np.random.seed(seed)

        # stats
        sarsa_episode_steps = []
        sarsa_episode_rewards = []
        sarsa_episode_success = []
        sarsa_episode_cliff_falls = []

        env = CliffWorld()
        state_shape = (env.rows, env.cols)
        num_actions = env.num_actions

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

                sarsa_agent.update(
                    state, action, new_reward, next_state, next_action, done
                )

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

        sarsa_all_episodes_steps.append(sarsa_episode_steps)
        sarsa_all_episodes_rewards.append(sarsa_episode_rewards)
        sarsa_all_episodes_success.append(sarsa_episode_success)
        sarsa_all_episodes_cliff_falls.append(sarsa_episode_cliff_falls)

    sarsa_all_episodes_rewards = np.array(sarsa_all_episodes_rewards)
    sarsa_all_episodes_steps = np.array(sarsa_all_episodes_steps)
    sarsa_all_episodes_success = np.array(sarsa_all_episodes_success)
    sarsa_all_episodes_cliff_falls = np.array(sarsa_all_episodes_cliff_falls)

    # check
    print(q_all_episodes_rewards.shape)
    print(sarsa_all_episodes_rewards.shape)

    print(q_all_episodes_success.mean())
    print(sarsa_all_episodes_success.mean())

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource")
    os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(save_dir, "q_learning_vs_sarsa_seeds.npz"),
        # experiment config
        seeds=seeds,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        num_episodes=num_episodes,
        max_steps=max_steps,
        # Q-learning
        q_rewards=q_all_episodes_rewards,
        q_steps=q_all_episodes_steps,
        q_success=q_all_episodes_success,
        q_cliff_falls=q_all_episodes_cliff_falls,
        # SARSA
        sarsa_rewards=sarsa_all_episodes_rewards,
        sarsa_steps=sarsa_all_episodes_steps,
        sarsa_success=sarsa_all_episodes_success,
        sarsa_cliff_falls=sarsa_all_episodes_cliff_falls,
    )
