import numpy as np

from bandit_lab.envs import BernoulliBandit, DriftingBernoulliBandit, GaussianBandit


def test_bernoulli_bandit_rewards_are_binary():
    env = BernoulliBandit(k=5, seed=0)
    rewards = [env.pull(0) for _ in range(50)]
    assert set(rewards).issubset({0.0, 1.0})


def test_bernoulli_bandit_optimal_arm_matches_probs():
    env = BernoulliBandit(k=4, probs=[0.1, 0.9, 0.3, 0.2], seed=0)
    assert env.optimal_arm == 1
    assert env.optimal_value == 0.9


def test_bernoulli_bandit_deterministic_with_seed():
    env_a = BernoulliBandit(k=5, seed=42)
    env_b = BernoulliBandit(k=5, seed=42)
    assert np.allclose(env_a.probs, env_b.probs)


def test_gaussian_bandit_optimal_arm_matches_means():
    env = GaussianBandit(k=3, means=[0.0, 5.0, -2.0], stds=[1.0, 1.0, 1.0], seed=0)
    assert env.optimal_arm == 1
    assert env.optimal_value == 5.0


def test_gaussian_bandit_pull_returns_float():
    env = GaussianBandit(k=3, means=[0.0, 1.0, 2.0], seed=0)
    assert isinstance(env.pull(0), float)


def test_drifting_bandit_probs_stay_in_unit_interval():
    env = DriftingBernoulliBandit(k=5, probs=[0.5] * 5, drift_std=0.2, seed=0)
    for _ in range(500):
        env.pull(0)
    assert np.all(env.probs >= 0.0) and np.all(env.probs <= 1.0)


def test_drifting_bandit_probs_actually_change():
    env = DriftingBernoulliBandit(k=5, probs=[0.5] * 5, drift_std=0.05, seed=0)
    initial = env.probs.copy()
    for _ in range(200):
        env.pull(0)
    assert not np.allclose(initial, env.probs)
