"""Gradient bandit algorithm (Sutton & Barto, 2018, section 2.8)."""
from __future__ import annotations

import numpy as np


class GradientBandit:
    def __init__(self, k: int, step_size: float = 0.1, use_baseline: bool = True, seed: int | None = None):
        self.k = k
        self.step_size = step_size
        self.use_baseline = use_baseline
        self.preferences = np.zeros(k, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.avg_reward = 0.0
        self.t = 0

    def _policy(self) -> np.ndarray:
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))
        return exp_prefs / exp_prefs.sum()

    def select_action(self) -> int:
        probs = self._policy()
        return int(self.rng.choice(self.k, p=probs))

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        baseline = self.avg_reward if self.use_baseline else 0.0
        probs = self._policy()
        one_hot = np.zeros(self.k)
        one_hot[arm] = 1.0
        self.preferences += self.step_size * (reward - baseline) * (one_hot - probs)
        if self.use_baseline:
            self.avg_reward += (reward - self.avg_reward) / self.t
