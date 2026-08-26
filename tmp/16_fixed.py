import numpy as np
from envs.cliff_world import CliffWorld
from agents.n_step_sarsa_agent import NStepSarsaAgent

if __name__ == "__main__":
    max_step = 200
    epsilon = 0.2
    gamma = 0.9
    alpha = 0.1
    n = 3

    env = CliffWorld()
    state_shape = (env.rows, env.cols)
    num_actions = env.num_actions

    agent = NStepSarsaAgent(
        state_shape, num_actions, n=n, alpha=alpha, gamma=gamma, epsilon=epsilon,
    )

    state = env.reset()
    action = agent.select_action(state)

    states = [state]
    actions = [action]
    rewards = []
    done = False

    for step in range(max_step):
        next_state, reward, done = env.step(action)
        rewards.append(reward)

        if not done:
            next_action = agent.select_action(next_state)
            states.append(next_state)
            actions.append(next_action)

        tau = step - n + 1
        if tau >= 0 and not done:
            G = 0.0
            for i in range(tau, tau + n):
                G += gamma ** (i - tau) * rewards[i]
            G += gamma**n * agent.q_table[states[tau + n]][actions[tau + n]]
            agent.update(states[tau], actions[tau], G)

        if done:
            T = len(rewards)
            start_tau = max(0, T - n)
            for flush_tau in range(start_tau, T):
                G = 0.0
                for i in range(flush_tau, T):
                    G += gamma ** (i - flush_tau) * rewards[i]
                agent.update(states[flush_tau], actions[flush_tau], G)
            break

        state = next_state
        action = next_action

    print()
    print("done:", done)
    print("num transitions:", len(rewards))
    print("states:", states)
    print("actions:", actions)
    print("rewards:", rewards)
