import random
import numpy as np
from gridworld import GridWorld

if __name__ == "__main__":

    env = GridWorld()
    state = env.reset()

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    max_step = 100
    num_episodes = 1000

    # q-table
    q_table = np.zeros((env.rows, env.cols, 4))

    for episode in range(num_episodes):
        state = env.reset()
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

            # print(
            #     f"step={step + 1}, "
            #     f"state={state}, "
            #     f"action={action}, "
            #     f"reward={reward}, "
            #     f"new_state={new_state}"
            # )

            if done:
                break

            state = new_state

    # evaluation
    state = env.reset()
    for step in range(max_step):
        row, col = state

        action = np.argmax(q_table[row, col])

        # execute action
        new_state, reward, done = env.step(action)

        env.render()

        if done:
            print(f"step: {step}")
            break

        state = new_state
