import os
import numpy as np
from gridworld import GridWorld

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resource")

if __name__ == "__main__":
    max_step = 100

    env = GridWorld()

    # load q_table
    q_table_path = os.path.join(RESOURCE_DIR, "q_table.npy")
    q_table = np.load(q_table_path)
    # print(q_table)

    # evaluation
    state = env.reset()
    for step in range(max_step):
        row, col = state

        action = np.argmax(q_table[row, col])

        # execute action
        new_state, reward, done = env.step(action)

        env.render()

        if done:
            print(f"step: {step + 1}")
            break

        state = new_state
