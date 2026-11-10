import numpy as np

from bandit_lab.agents import EpsilonGreedy
from bandit_lab.envs import BernoulliBandit
from bandit_lab.metrics import run_bandit


def test_run_bandit_output_shapes():
    env = BernoulliBandit(k=5, seed=0)
    agent = EpsilonGreedy(k=5, epsilon=0.1, seed=0)
    result = run_bandit(env, agent, steps=100)
    assert result["rewards"].shape == (100,)
    assert result["chosen_optimal"].shape == (100,)
    assert result["cumulative_regret"].shape == (100,)
    assert result["optimal_action_rate"].shape == (100,)


def test_cumulative_regret_is_nondecreasing():
    env = BernoulliBandit(k=5, seed=0)
    agent = EpsilonGreedy(k=5, epsilon=0.1, seed=0)
    result = run_bandit(env, agent, steps=200)
    diffs = np.diff(result["cumulative_regret"])
    assert np.all(diffs >= -1e-9)


def test_optimal_action_rate_bounded_in_unit_interval():
    env = BernoulliBandit(k=5, seed=0)
    agent = EpsilonGreedy(k=5, epsilon=0.1, seed=0)
    result = run_bandit(env, agent, steps=200)
    assert np.all(result["optimal_action_rate"] >= 0.0)
    assert np.all(result["optimal_action_rate"] <= 1.0)


def test_always_optimal_agent_has_zero_regret():
    env = BernoulliBandit(k=3, probs=[0.1, 0.9, 0.2], seed=0)

    class AlwaysBest:
        def select_action(self):
            return 1

        def update(self, arm, reward):
            pass

    result = run_bandit(env, AlwaysBest(), steps=50)
    assert np.allclose(result["cumulative_regret"], 0.0)
    assert np.all(result["chosen_optimal"])
