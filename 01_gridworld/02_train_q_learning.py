import random
import numpy as np
import os
from gridworld import GridWorld

if __name__ == "__main__":

    env = GridWorld()

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    max_step = 50
    num_episodes = 20

    # q-table
    q_table = np.zeros((env.rows, env.cols, 4))

    episode_rewards = []
    episode_steps = []
    episode_success = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_step):
            row, col = state

            # epsilon-greedy
            random_value = random.random()
            if random_value < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_table[row, col])

            # execute action
            new_state, reward, done = env.step(action)
            old_q = q_table[row, col, action]

            total_reward += reward
            # if agent arrive at Goal, game over, reward is target
            # if not, game continue
            if done:
                target = reward
            else:
                next_row, next_col = new_state
                best_future_q = np.max(q_table[next_row, next_col])
                target = reward + gamma * best_future_q
            new_q = old_q + alpha * (target - old_q)
            q_table[row, col, action] = new_q

            if done:
                episode_steps.append(step + 1)
                break

            state = new_state

        episode_rewards.append(total_reward)
        if not done:
            episode_steps.append(max_step)
        episode_success.append(done)
    # save q-table to resource folder (create it if not exists)
    save_dir = os.path.join(os.path.dirname(__file__), "resource")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "q_table.npy"), q_table)

    window_size = 5
    for start in range(0, num_episodes, window_size):
        # stats - all
        success_rate = np.mean(episode_success[start : start + window_size])

        average_reward = np.mean(episode_rewards[start : start + window_size])

        average_steps = np.mean(episode_steps[start : start + window_size])

        # stats - only success episode: filter indices where episode_success is True
        success_indices = [
            i
            for i, success in enumerate(episode_success[start : start + window_size])
            if success
        ]

        average_reward_success = np.mean(
            [episode_rewards[i + start] for i in success_indices]
        )
        average_steps_success = np.mean(
            [episode_steps[i + start] for i in success_indices]
        )

        print(
            f"Episodes: {start}-{start + window_size}",
            f"success_rate: {success_rate}",
            f"average_reward: {average_reward}",
            f"average_steps: {average_steps}",
            f"average_reward_success: {average_reward_success}",
            f"average_steps_success: {average_steps_success}",
        )
