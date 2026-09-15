import numpy as np


class SarsaLambda:
    def __int__(
        self,
        state_shape,
        num_actions,
        n=3,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2,
        seed=None,
    ):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.q_table = np.zeros((*state_shape, num_actions))

        self.rng = np.random.default_rng(seed)
