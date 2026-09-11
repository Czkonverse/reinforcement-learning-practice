from envs.cliff_world import CliffWorld
from agents.n_step_sarsa_agent import NStepSarsaAgent

if __name__ == "__main__":
    num_episodes = 200
    max_step = 200
    n = 3
    epsilon = 0.2
    gamma = 0.9
    alpha = 0.1
    eval_every = 20

    env = CliffWorld()
    agent = NStepSarsaAgent(
        (env.rows, env.cols),
        env.num_actions,
        n=n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )
