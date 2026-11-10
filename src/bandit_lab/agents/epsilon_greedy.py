"""Epsilon-greedy action-value agent with sample-average or constant step size."""
from __future__ import annotations

import numpy as np


class EpsilonGreedy:
    def __init__(
        self,
        k: int,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        step_size: float | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            epsilon: probability of picking a uniformly random arm instead of greedy.
            initial_value: optimistic (or pessimistic) initial Q estimate for every arm.
            step_size: if ``None``, use the sample-average update (1/N(a)); if a float,
                use a constant step size alpha, which weights recent rewards more
                heavily and is appropriate for non-stationary bandits.
        """
        self.k = k
        self.epsilon = epsilon
        self.step_size = step_size
        self.rng = np.random.default_rng(seed)
        self.q = np.full(k, initial_value, dtype=float)
        self.n = np.zeros(k, dtype=int)

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        return int(np.argmax(self.q))

    def update(self, arm: int, reward: float) -> None:
        self.n[arm] += 1
        alpha = self.step_size if self.step_size is not None else 1.0 / self.n[arm]
        self.q[arm] += alpha * (reward - self.q[arm])
