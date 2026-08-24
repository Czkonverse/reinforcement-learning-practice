import numpy as np
from envs.cliff_world import CliffWorld
from agents.n_step_sarsa_agent import NStepSarsaAgent

# S0 --A0--> S1 --A1--> S2 --A2--> S3--A3--> Terminal
#      R1         R2         R3        R4

# update, Q(S0​,A0​) --> R1​+γR2​+γ2R3​+γ3Q(S3​,A3​)
if __name__ == "__main__":

    max_step = 200
    epsilon = 0.2
    gamma = 0.9
    alpha = 0.1

    n = 3
    tau = 0

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
            states.append(next_state)
            next_action = agent.select_action(next_state)
            actions.append(next_action)

        tau = step - n + 1
        if tau >= 0:
            if not done:
                old_state = states[tau]
                old_action = actions[tau]
                old_q = agent.q_table[old_state][old_action]

                G = 0
                for i in range(tau, tau + n):
                    G = G + gamma ** (i - tau) * rewards[i]
                G = G + gamma**n * agent.q_table[next_state][next_action]

                new_q = old_q + alpha * (G - old_q)
                agent.q_table[old_state][old_action] = new_q
            else:
                G = 0
                for i in range(tau, n + 1):
                    state = states[i]
                    action = actions[i]

                    q_old = agent.q_table[state][action]

                    for j in range(0, i):
                        G = G + gamma * (i)

        action = next_action
        state = next_state
