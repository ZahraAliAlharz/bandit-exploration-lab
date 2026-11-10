"""Upper-Confidence-Bound (UCB1) agent."""
from __future__ import annotations

import numpy as np


class UCB1:
    def __init__(self, k: int, c: float = 2.0):
        """
        Args:
            c: exploration weight on the confidence term
               ``c * sqrt(ln(t) / N(a))`` (Auer, Cesa-Bianchi & Fischer, 2002).
        """
        self.k = k
        self.c = c
        self.q = np.zeros(k, dtype=float)
        self.n = np.zeros(k, dtype=int)
        self.t = 0

    def select_action(self) -> int:
        # Play every arm once before trusting the confidence bound.
        unplayed = np.where(self.n == 0)[0]
        if len(unplayed) > 0:
            return int(unplayed[0])
        bonus = self.c * np.sqrt(np.log(self.t) / self.n)
        return int(np.argmax(self.q + bonus))

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]
