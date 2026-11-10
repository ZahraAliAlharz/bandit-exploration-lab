"""Thompson sampling with a Beta-Bernoulli conjugate posterior."""
from __future__ import annotations

import numpy as np


class ThompsonSamplingBeta:
    def __init__(self, k: int, prior_alpha: float = 1.0, prior_beta: float = 1.0, seed: int | None = None):
        self.k = k
        self.alpha = np.full(k, prior_alpha, dtype=float)
        self.beta = np.full(k, prior_beta, dtype=float)
        self.rng = np.random.default_rng(seed)

    def select_action(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        # Reward is assumed Bernoulli in {0, 1}; a Beta(1,1) prior with this
        # update is the exact posterior for a Bernoulli bandit.
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward
