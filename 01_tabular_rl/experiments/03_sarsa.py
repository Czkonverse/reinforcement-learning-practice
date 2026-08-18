import numpy as np
from envs.grid_world import GridWorld
from agents.sarsa_agent import SarsaAgent


def evaluate(env, agent, max_step):
    state = env.reset()

    total_rewards = 0
    for step in range(max_step):

        action = agent.select_action(state, training=False)
        next_state, reward, done = env.step(action)

        total_rewards += reward

        if done:
            return done, step + 1, total_rewards

        state = next_state

    return False, max_step, total_rewards


if __name__ == "__main__":

    num_episodes = 80
    max_step = 80
    epsilon = 0.2
    gamma = 0.9
    alpha = 0.1

    env = GridWorld()
    state_shape = (env.rows, env.cols)
    num_actions = env.num_acitons
    agent = SarsaAgent(state_shape, num_actions, alpha, gamma, epsilon)

    episode_steps = []
    episode_rewards = []
    episode_success = []
    # step=0, initiate action
    for episode in range(num_episodes):
        state = env.reset()
        action = agent.select_action(state)

        total_reward = 0
        done = False
        for step in range(max_step):

            next_state, new_reward, done = env.step(action)
            total_reward += new_reward

            next_action = agent.select_action(next_state)

            agent.update(state, action, new_reward, next_state, next_action, done)

            if done:
                episode_steps.append(step + 1)
                break

            action = next_action
            state = next_state

        if not done:
            episode_steps.append(max_step)

        episode_success.append(done)
        episode_rewards.append(total_reward)
        success_rate = np.mean(episode_success)

    print("avg episode_steps: ", np.mean(episode_steps))
    print("avg episode_rewards: ", np.mean(episode_rewards))
    print("success_rate: ", success_rate)

    # evaluate
    test_env = GridWorld()
    eval_done, eval_max_step, eval_total_rewards = evaluate(env, agent, max_step)
    print(
        "eval_done, eval_max_step, eval_total_rewards: \n",
        eval_done,
        eval_max_step,
        eval_total_rewards,
    )
