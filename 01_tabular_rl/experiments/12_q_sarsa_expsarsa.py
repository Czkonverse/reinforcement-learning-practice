import os

import numpy as np
from envs.cliff_world import CliffWorld
from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SarsaAgent
from agents.exp_sarsa_agent import ExpectedSarsaAgent


def train_expected_sarsa(
    env,
    seed,
    epsilon,
    alpha=0.1,
    gamma=0.9,
    num_episodes=300,
    max_steps=200,
    cliff_falls_reward=-100,
):
    np.random.seed(seed)

    # stats
    episode_steps = []
    episode_rewards = []
    episode_success = []
    episode_cliff_falls = []

    state_shape = (env.rows, env.cols)
    num_actions = env.num_actions

    agent = ExpectedSarsaAgent(state_shape, num_actions, alpha, gamma, epsilon)

    for episode in range(num_episodes):

        state = env.reset()

        total_rewards = 0
        cliff_falls = 0
        done = False
        for step in range(max_steps):

            action = agent.select_action(state)

            next_state, reward, done = env.step(action)
            total_rewards += reward
            if reward == cliff_falls_reward:
                cliff_falls += 1

            agent.update(state, action, reward, next_state, done)

            if done:
                episode_steps.append(step + 1)
                break

            state = next_state

        if not done:
            episode_steps.append(max_steps)
        episode_rewards.append(total_rewards)
        episode_success.append(done)
        episode_cliff_falls.append(cliff_falls)

    return episode_steps, episode_rewards, episode_success, episode_cliff_falls


def train_sarsa(
    env,
    seed,
    epsilon,
    alpha=0.1,
    gamma=0.9,
    num_episodes=300,
    max_steps=200,
    cliff_falls_reward=-100,
):
    np.random.seed(seed)

    # stats
    sarsa_episode_steps = []
    sarsa_episode_rewards = []
    sarsa_episode_success = []
    sarsa_episode_cliff_falls = []

    # sarsa
    state_shape = (env.rows, env.cols)
    num_actions = env.num_actions

    sarsa_agent = SarsaAgent(state_shape, num_actions, alpha, gamma, epsilon)

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

            if done:
                sarsa_agent.update(
                    state,
                    action,
                    new_reward,
                    next_state,
                    None,
                    done,
                )

                sarsa_episode_steps.append(step + 1)
                break

            next_action = sarsa_agent.select_action(next_state)

            sarsa_agent.update(state, action, new_reward, next_state, next_action, done)

            action = next_action
            state = next_state

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
    )


def train_q_learning(
    env,
    seed,
    epsilon,
    alpha=0.1,
    gamma=0.9,
    num_episodes=300,
    max_steps=200,
    cliff_falls_reward=-100,
):
    np.random.seed(seed)

    # stats
    q_episode_steps = []
    q_episode_rewards = []
    q_episode_success = []
    q_episode_cliff_falls = []

    # q learning
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

    return q_episode_steps, q_episode_rewards, q_episode_success, q_episode_cliff_falls


if __name__ == "__main__":

    alpha = 0.1
    gamma = 0.9
    num_episodes = 1000
    max_steps = 200
    cliff_falls_reward = -100

    seeds = np.arange(20)

    # epsilons = [0.05, 0.1, 0.2, 0.3]
    epsilons = [0.2]

    env = CliffWorld()

    # stats - epsilon
    for epsilon in epsilons:
        # stats - all episodes
        q_all_episodes_steps = []
        q_all_episodes_rewards = []
        q_all_episodes_success = []
        q_all_episodes_cliff_falls = []

        sarsa_all_episodes_steps = []
        sarsa_all_episodes_rewards = []
        sarsa_all_episodes_success = []
        sarsa_all_episodes_cliff_falls = []

        exp_sarsa_all_episodes_steps = []
        exp_sarsa_all_episodes_rewards = []
        exp_sarsa_all_episodes_success = []
        exp_sarsa_all_episodes_cliff_falls = []

        for seed in seeds:
            ########################## Q Learning ##########################
            (
                q_episode_steps,
                q_episode_rewards,
                q_episode_success,
                q_episode_cliff_falls,
            ) = train_q_learning(
                env,
                seed,
                epsilon,
                alpha,
                gamma,
                num_episodes,
                max_steps,
                cliff_falls_reward,
            )

            q_all_episodes_steps.append(q_episode_steps)
            q_all_episodes_rewards.append(q_episode_rewards)
            q_all_episodes_success.append(q_episode_success)
            q_all_episodes_cliff_falls.append(q_episode_cliff_falls)

            ########################## SARSA ##########################
            (
                sarsa_episode_steps,
                sarsa_episode_rewards,
                sarsa_episode_success,
                sarsa_episode_cliff_falls,
            ) = train_sarsa(
                env,
                seed,
                epsilon,
                alpha,
                gamma,
                num_episodes,
                max_steps,
                cliff_falls_reward,
            )

            sarsa_all_episodes_steps.append(sarsa_episode_steps)
            sarsa_all_episodes_rewards.append(sarsa_episode_rewards)
            sarsa_all_episodes_success.append(sarsa_episode_success)
            sarsa_all_episodes_cliff_falls.append(sarsa_episode_cliff_falls)

            ###################### Expected SARSA #######################
            (
                exp_sarsa_episode_steps,
                exp_sarsa_episode_rewards,
                exp_sarsa_episode_success,
                exp_sarsa_episode_cliff_falls,
            ) = train_expected_sarsa(
                env,
                seed,
                epsilon,
                alpha,
                gamma,
                num_episodes,
                max_steps,
                cliff_falls_reward,
            )

            exp_sarsa_all_episodes_steps.append(exp_sarsa_episode_steps)
            exp_sarsa_all_episodes_rewards.append(exp_sarsa_episode_rewards)
            exp_sarsa_all_episodes_success.append(exp_sarsa_episode_success)
            exp_sarsa_all_episodes_cliff_falls.append(exp_sarsa_episode_cliff_falls)

        q_all_episodes_rewards = np.array(q_all_episodes_rewards)
        q_all_episodes_steps = np.array(q_all_episodes_steps)
        q_all_episodes_success = np.array(q_all_episodes_success)
        q_all_episodes_cliff_falls = np.array(q_all_episodes_cliff_falls)

        sarsa_all_episodes_rewards = np.array(sarsa_all_episodes_rewards)
        sarsa_all_episodes_steps = np.array(sarsa_all_episodes_steps)
        sarsa_all_episodes_success = np.array(sarsa_all_episodes_success)
        sarsa_all_episodes_cliff_falls = np.array(sarsa_all_episodes_cliff_falls)

        exp_sarsa_all_episodes_rewards = np.array(exp_sarsa_all_episodes_rewards)
        exp_sarsa_all_episodes_steps = np.array(exp_sarsa_all_episodes_steps)
        exp_sarsa_all_episodes_success = np.array(exp_sarsa_all_episodes_success)
        exp_sarsa_all_episodes_cliff_falls = np.array(
            exp_sarsa_all_episodes_cliff_falls
        )

        # check
        for arr in [
            q_all_episodes_rewards,
            q_all_episodes_steps,
            q_all_episodes_success,
            q_all_episodes_cliff_falls,
            sarsa_all_episodes_rewards,
            sarsa_all_episodes_steps,
            sarsa_all_episodes_success,
            sarsa_all_episodes_cliff_falls,
            exp_sarsa_all_episodes_rewards,
            exp_sarsa_all_episodes_steps,
            exp_sarsa_all_episodes_success,
            exp_sarsa_all_episodes_cliff_falls,
        ]:
            assert arr.shape == (len(seeds), num_episodes)

        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource")
        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(
                save_dir, f"q_sarsa_expsarsa_eps-{epsilon}_seeds-{len(seeds)}.npz"
            ),
            # experiment config
            seeds=seeds,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            num_episodes=num_episodes,
            max_steps=max_steps,
            cliff_falls_reward=cliff_falls_reward,
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
            # Expected SARSA
            exp_sarsa_rewards=exp_sarsa_all_episodes_rewards,
            exp_sarsa_steps=exp_sarsa_all_episodes_steps,
            exp_sarsa_success=exp_sarsa_all_episodes_success,
            exp_sarsa_cliff_falls=exp_sarsa_all_episodes_cliff_falls,
        )
