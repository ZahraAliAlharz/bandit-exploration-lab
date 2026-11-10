"""Multi-armed bandit environments.

Each environment exposes ``k`` arms and a ``pull(arm)`` method returning a
scalar reward. ``optimal_arm`` reports whichever arm currently has the
highest expected reward, so callers can measure regret without the agent
needing access to it.
"""
from __future__ import annotations

import numpy as np


class BernoulliBandit:
    """k-armed bandit; arm ``i`` pays 1 with probability ``probs[i]``, else 0."""

    def __init__(self, k: int, probs: np.ndarray | None = None, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.probs = self.rng.uniform(0.0, 1.0, size=k) if probs is None else np.asarray(probs, dtype=float)

    def pull(self, arm: int) -> float:
        return float(self.rng.random() < self.probs[arm])

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def optimal_value(self) -> float:
        return float(np.max(self.probs))

    def expected_value(self, arm: int) -> float:
        return float(self.probs[arm])


class GaussianBandit:
    """k-armed bandit; arm ``i`` pays ``N(means[i], stds[i])``."""

    def __init__(
        self,
        k: int,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.means = self.rng.normal(0.0, 1.0, size=k) if means is None else np.asarray(means, dtype=float)
        self.stds = np.ones(k) if stds is None else np.asarray(stds, dtype=float)

    def pull(self, arm: int) -> float:
        return float(self.rng.normal(self.means[arm], self.stds[arm]))

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self.means))

    @property
    def optimal_value(self) -> float:
        return float(np.max(self.means))

    def expected_value(self, arm: int) -> float:
        return float(self.means[arm])


class DriftingBernoulliBandit(BernoulliBandit):
    """Non-stationary Bernoulli bandit.

    After every pull, every arm's success probability takes an independent
    Gaussian random-walk step and is clipped back into ``[0, 1]``. This is
    the "restless bandit" setting from Sutton & Barto (2018), exercise 2.5,
    used to show that recency-weighted (constant step-size) value estimates
    track a moving target better than unweighted sample averages.
    """

    def __init__(
        self,
        k: int,
        probs: np.ndarray | None = None,
        drift_std: float = 0.01,
        seed: int | None = None,
    ):
        super().__init__(k, probs, seed)
        self.drift_std = drift_std

    def pull(self, arm: int) -> float:
        reward = super().pull(arm)
        self.probs = np.clip(self.probs + self.rng.normal(0.0, self.drift_std, size=self.k), 0.0, 1.0)
        return reward
