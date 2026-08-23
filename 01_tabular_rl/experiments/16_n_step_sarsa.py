import numpy as np
from envs.cliff_world import CliffWorld
from agents.n_step_sarsa_agent import NStepSarsaAgent

if __name__ == "__main__":

    epsilon = 0.2
    gamma = 0.9
    alpha = 0.1

    n = 3
    tao = 0

    env = CliffWorld()
    state_shape = (env.rows, env.cols)
    num_actions = env.num_acitons

    agent = NStepSarsaAgent(
        state_shape,
        num_actions,
        n=n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )

    # start
    state = env.reset()
    action = agent.select_action(state)

    states = [state]
    actions = [action]
    rewards = []

    for step in range(max_step):
        next_state, new_reward, done = env.step(action)
        rewards.append(new_reward)

        if not done:
            next_action = agent.select_action(next_state)
            states.append(next_state)
            actions.append(next_action)

        if done:
            pass

        tau = step - n + 1
        if tao >= 0:

            G = (
                rewards[-3]
                + gamma * rewards[-2]
                + (gamma**2) * rewards[-1]
                + (gamma**3) * agent.q_table[next_state][next_action]
            )

            old_q = agent.q_table[states[-3]][actions[-3]]
            agent.q_table[states[-3]][actions[-3]] = old_q + alpha * (G - old_q)

        else:
            pass

        action = next_action
        state = next_state
