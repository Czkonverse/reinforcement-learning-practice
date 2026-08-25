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

        # update
        tau = step - n + 1
        G = 0

        # 1 tau<0, done=False, no update;
        # 2 tau<0, done=True, flush;
        # 3 tau>=0, done=False, update;
        # 3 tau>=0, done=True, flush;
        if tau >= 0 and not done:
            old_state = states[tau]
            old_action = actions[tau]
            old_q = agent.q_table[old_state][old_action]

            for i in range(tau, tau + n):
                G = G + gamma ** (i - tau) * rewards[i]
            G = G + gamma**n * agent.q_table[next_state][next_action]
            agent.update(
                states[tau],
                actions[tau],
                G,
            )

            # terminal
            if done:
                break

            state = next_state
            action = next_action

        # tau = t - n + 1, t------index of the transition
        # T - total transition
        if done:
            T = len(rewards)

            start_tau = max(0, T - n)
            for t in range(start_tau, T):
                G = 0.0
                for i in range(t, T):
                    G += gamma ** (i - tau) * rewards[i]

                agent.update(
                    states[tau],
                    actions[tau],
                    G,
                )
