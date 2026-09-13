"""Verify n_step_sarsa_agent against an independent reference implementation.

Actions are forced to be random (epsilon=1) so they do not depend on Q; the
agent's own recorded trace is then replayed by a from-scratch reference.
"""

import sys
import numpy as np

sys.path.insert(0, "01_tabular_rl")
from agents.n_step_sarsa_agent import NStepSarsaAgent


def run_reference(states, actions, rewards, n, alpha, gamma, mode):
    """Independent n-step SARSA on a fixed trace.

    states  : S_0 .. S_T          (len T+1)
    actions : A_0 .. A_{T-1} (terminal) or A_0 .. A_T (truncated)
    rewards : R_1 .. R_T          (len T, rewards[i] == R_{i+1})
    """
    T = len(rewards)
    q = {}
    get = lambda s, a: q.get((s, a), 0.0)  # noqa: E731

    def upd(s, a, target):
        old = q.get((s, a), 0.0)
        q[(s, a)] = old + alpha * (target - old)

    # in-episode n-step updates happen on every non-final step (t = 0 .. T-2)
    for t in range(T - 1):
        tau = t - n + 1
        if tau >= 0:
            G = sum(gamma ** (i - tau) * rewards[i] for i in range(tau, tau + n))
            G += gamma**n * get(states[tau + n], actions[tau + n])
            upd(states[tau], actions[tau], G)

    # flush tail
    for tau in range(max(0, T - n), T):
        G = sum(gamma ** (i - tau) * rewards[i] for i in range(tau, T))
        if mode == "truncated":
            G += gamma ** (T - tau) * get(states[T], actions[T])
        upd(states[tau], actions[tau], G)

    return q


def q_to_array(agent):
    return agent.q_table.copy()


def test_mode(mode, n, T, shape, num_actions, seed):
    alpha, gamma = 0.3, 0.9
    agent = NStepSarsaAgent(
        state_shape=shape,
        num_actions=num_actions,
        n=n,
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1)

    s0 = (0, 0)
    a0 = agent.begin_episode(s0)
    states = [s0]
    actions = [a0]
    rewards = list(rng.integers(-3, 4, size=T))  # R_1 .. R_T

    for t in range(T):
        ns = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
        states.append(ns)
        terminal = (t == T - 1) and mode == "terminal"
        truncated = (t == T - 1) and mode == "truncated"
        a = agent.step(int(rewards[t]), ns, terminal, truncated)
        if not terminal:
            actions.append(a)

    # snapshot the trace before buffers are cleared
    states_copy = list(agent._states)
    actions_copy = list(agent._actions)
    rewards_copy = list(agent._rewards)
    assert states_copy == states, "recorded states differ"
    assert rewards_copy == rewards, "recorded rewards differ"

    agent.end_episode(mode)  # clears buffers
    assert agent._states == [] and agent._actions == [] and agent._rewards == []

    q_agent = q_to_array(agent)
    q_ref = run_reference(
        states_copy, actions_copy, rewards_copy, n, alpha, gamma, mode
    )

    # compare every cell
    for si in range(shape[0]):
        for sj in range(shape[1]):
            for ai in range(num_actions):
                exp = q_ref.get(((si, sj), ai), 0.0)
                got = q_agent[si, sj, ai]
                assert abs(exp - got) < 1e-9, (
                    f"[{mode} n={n} T={T}] Q({si},{sj},{ai}) "
                    f"expected {exp}, got {got}"
                )
    return True


def test_coverage(mode, n, T, seed):
    """Every transition tau in [0, T-1] must be updated exactly once."""
    agent = NStepSarsaAgent(
        state_shape=(2, 2),
        num_actions=4,
        n=n,
        alpha=0.3,
        gamma=0.9,
        epsilon=1.0,
        seed=seed,
    )
    taus = []
    orig = agent._update_Q

    def spy(tau, target):
        taus.append(tau)
        orig(tau, target)

    agent._update_Q = spy
    rng = np.random.default_rng(seed)
    agent.begin_episode((0, 0))
    for t in range(T):
        ns = (int(rng.integers(2)), int(rng.integers(2)))
        terminal = (t == T - 1) and mode == "terminal"
        truncated = (t == T - 1) and mode == "truncated"
        agent.step(1, ns, terminal, truncated)
    agent.end_episode(mode)

    assert sorted(taus) == list(
        range(T)
    ), f"[{mode} n={n} T={T}] updated taus = {sorted(taus)}, expected {list(range(T))}"
    return True


if __name__ == "__main__":
    ok = 0
    for mode in ("terminal", "truncated"):
        for n in (1, 2, 3, 5):
            for T in (1, 2, 3, 5, 8, 12):
                test_mode(mode, n, T, (3, 3), 4, seed=100 * n + T)
                test_coverage(mode, n, T, seed=100 * n + T)
                ok += 1
    print(f"ALL PASSED ({ok} configs x2 checks)")

    # smoke test on the real env
    from envs.cliff_world import CliffWorld

    for mode_seed in range(3):
        env = CliffWorld()
        agent = NStepSarsaAgent((4, 12), 4, n=3, seed=mode_seed)
        state = env.reset()
        action = agent.begin_episode(state)
        done = False
        for _ in range(100):
            state, reward, done = env.step(action)
            action = agent.step(reward, state, done, truncated=False)
            if done:
                break
        agent.end_episode("terminal" if done else "truncated")
    print("CLIFFWORLD SMOKE TEST PASSED")
