import numpy as np


class MaximizationBiasEnv:
    def __init__(self, reward_mean=-0.1, reward_std=1.0, seed=None):
        self.reward_mean = reward_mean
        self.reward_std = reward_std

        self.rng = np.random.default_rng(seed)

        # states
        self.A = 0
        self.B = 1
        self.TERMINAL = 2

        self.state_actions = {self.A: (0, 1), self.B: tuple(range(10))}

        self.state = self.A

    def reset(self):
        self.state = self.A
        return self.state

    def get_valid_actions(self, state=None):
        if state is None:
            state = self.state

        if state == self.TERMINAL:
            return ()

        if state not in self.state_actions:
            raise ValueError(f"Invalid state: {state}")

        return self.state_actions[state]

    def step(self, action):
        if self.state == self.TERMINAL:
            raise RuntimeError(
                "Episode has already terminated. Call reset() before step()."
            )

        valid_actions = self.get_valid_actions()

        if action not in valid_actions:
            raise ValueError(
                f"Invalid action {action} for state {self.state}. "
                f"Valid actions: {valid_actions}"
            )

        # state A
        if self.state == self.A:
            if action == 0:
                next_state = self.TERMINAL
                reward = 0.0
                done = True
            else:
                next_state = self.B
                reward = 0.0
                done = False
        elif self.state == self.B:
            next_state = self.TERMINAL

            reward = self.rng.normal(loc=self.reward_mean, scale=self.reward_std)
            done = True

        self.state = next_state

        return next_state, reward, done


if __name__ == "__main__":
    env = MaximizationBiasEnv()

    next_state, reward, done = env.step(action=0)
    print(next_state, reward, done)

    state = env.reset()
    next_state, reward, done = env.step(action=1)
    print(next_state, reward, done)

    random_num = np.random.randint(10)
    print(random_num)
    next_state, reward, done = env.step(action=random_num)
    print(next_state, reward, done)

    state = env.reset()
    next_state, reward, done = env.step(action=1)
    print(next_state, reward, done)

    test_times = 1000
    total_rewards = 0
    for i in range(test_times):
        state = env.reset()
        next_state, reward, done = env.step(action=1)
        random_num = np.random.randint(10)
        next_state, reward, done = env.step(action=random_num)
        total_rewards = total_rewards + reward
    print(total_rewards, test_times)
    print(total_rewards / test_times)

q_a = np.zeros(2)
q_b = np.zeros(10)
