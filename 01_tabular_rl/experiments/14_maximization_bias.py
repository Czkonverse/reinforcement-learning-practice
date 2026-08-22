import numpy as np
from envs.maximization_bias import MaximizationBiasEnv
from agents.q_learning_agent_v2 import QLearningAgentV2


def train_q_learning(
    num_seeds=20,
    num_episodes=300,
    alpha=0.1,
    gamma=1.0,
    epsilon=0.1,
):
    # shape:
    # (num_seeds, num_episodes)
    #
    # 0: action 0 at A -> Terminal
    # 1: action 1 at A -> B
    entered_b_records = np.zeros(
        (num_seeds, num_episodes),
        dtype=int,
    )

    for seed in range(num_seeds):

        # 每个 seed 都重新创建环境和 agent
        # 因此每次实验都从全新的 Q-table 开始
        env = MaximizationBiasEnv(
            reward_mean=-0.1,
            reward_std=1.0,
            seed=seed,
        )

        agent = QLearningAgentV2(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            seed=seed + 10000,
        )

        for episode in range(num_episodes):

            state = env.reset()

            done = False

            while not done:

                action = agent.select_action(state)

                # 我们只关心在 A 时选择了什么
                if state == env.A:
                    if action == 1:
                        entered_b_records[seed, episode] = 1
                    else:
                        entered_b_records[seed, episode] = 0

                next_state, reward, done = env.step(action)

                agent.update(
                    state=state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                    done=done,
                )

                state = next_state

    return entered_b_records


if __name__ == "__main__":

    num_seeds = 20
    num_episodes = 300

    entered_b_records = train_q_learning(
        num_seeds=num_seeds,
        num_episodes=num_episodes,
        alpha=0.1,
        gamma=1.0,
        epsilon=0.1,
    )

    print("shape:", entered_b_records.shape)

    # 每一个 episode，在 20 个 seeds 中有多少比例选择进入 B
    right_action_rate = np.mean(entered_b_records, axis=0)

    print("First 20 episodes:")
    print(right_action_rate[:20])

    print()
    print("Last 20 episodes:")
    print(right_action_rate[-20:])

    print()
    print(
        "Average right-action rate over all episodes:",
        np.mean(right_action_rate),
    )
