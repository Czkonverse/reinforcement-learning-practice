import numpy as np
import random
from envs.gridworld import GridWorld


def evaluate(env, q_table, max_step):
    state = env.reset()

    total_rewards = 0
    for step in range(max_step):

        action = greedy_action(q_table, state)
        new_state, reward, done = env.step(action)

        total_rewards += reward

        if done:
            return done, step + 1, total_rewards

        state = new_state

    return False, max_step, total_rewards


def epsilon_greedy_action(q_table, state, epsilon):

    if random.random() < epsilon:
        action = random.randint(0, q_table.shape[-1] - 1)
    else:
        action = greedy_action(q_table, state)

    return action


def greedy_action(q_table, state):
    row, col = state
    q_values = q_table[row, col]

    max_q = np.max(q_values)
    best_actions = np.flatnonzero(q_values == max_q)

    return np.random.choice(best_actions)


if __name__ == "__main__":

    num_episodes = 80
    max_steps = 80
    epsilon = 0.2
    gamma = 0.9
    alpha = 0.1

    env = GridWorld()

    q_table = np.zeros((env.rows, env.cols, 4))

    episode_steps = []
    episode_rewards = []
    episode_success = []
    # step=0, initiate action
    for episode in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy_action(q_table, state, epsilon)

        total_reward = 0
        done = False
        for step in range(max_steps):
            row, col = state
            q_old = q_table[row, col, action]

            new_state, new_reward, done = env.step(action)
            total_reward += new_reward

            if done:
                target = new_reward
            else:
                new_action = epsilon_greedy_action(q_table, new_state, epsilon)

                new_row, new_col = new_state
                q_new = q_table[new_row, new_col, new_action]

                target = new_reward + gamma * q_new

            q_table[row, col, action] = q_old + alpha * (target - q_old)

            if done:
                episode_steps.append(step + 1)
                break

            action = new_action
            state = new_state

        if not done:
            episode_steps.append(max_steps)

        episode_success.append(done)
        episode_rewards.append(total_reward)
        success_rate = np.mean(episode_success)

    print("avg episode_steps: ", np.mean(episode_steps))
    print("avg episode_rewards: ", np.mean(episode_rewards))
    print("success_rate: ", success_rate)

    # evaluate
    test_env = GridWorld()
    eval_done, eval_max_step, eval_total_rewards = evaluate(
        test_env, q_table, max_steps
    )
    print(
        "eval_done, eval_max_step, eval_total_rewards: \n",
        eval_done,
        eval_max_step,
        eval_total_rewards,
    )
