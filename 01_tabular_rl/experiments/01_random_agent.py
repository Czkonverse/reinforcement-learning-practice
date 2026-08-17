from envs.gridworld import GridWorld
from agents.random_agent import RandomAgent

if __name__ == "__main__":
    env = GridWorld()

    num_actions = 4
    agent = RandomAgent(num_actions)

    num_episodes = 1000
    max_step = 100

    success_count = 0
    total_success_steps = 0

    for episode in range(num_episodes):
        state = env.reset()

        done = False
        for step in range(max_step):
            # state - current
            # action
            action = agent.select_action()
            # execute action
            new_state, reward, done = env.step(action)
            # print("new_state, reward, done: ", new_state, reward, done, "\n")

            if done:
                # print(f"Finish at step: {step+1}!!! \n")
                success_count += 1
                total_success_steps = total_success_steps + step + 1
                # env.render()
                break

        # if not done:
        # print(f"After max step {max_step}, agent can't find the exit......")

    # final statiscs
    success_rate = success_count / num_episodes
    if success_count > 0:
        average_steps = total_success_steps / success_count
    else:
        average_steps = 0

    print(f"Episodes: {num_episodes}")
    print(f"Success: {success_count}")
    print(f"Failed: {num_episodes - success_count}")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Average success steps: {average_steps:.2f}")
