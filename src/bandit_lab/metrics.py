"""Shared metrics for bandit runs: regret and optimal-action rate."""
from __future__ import annotations

import numpy as np


def run_bandit(env, agent, steps: int) -> dict:
    """Run ``agent`` against ``env`` for ``steps`` pulls.

    Returns per-step arrays so callers can compute cumulative regret,
    optimal-action rate, or any other quantity of interest, including for
    non-stationary environments where the optimal arm can change mid-run.
    """
    rewards = np.zeros(steps)
    chosen_optimal = np.zeros(steps, dtype=bool)
    instant_regret = np.zeros(steps)

    for t in range(steps):
        optimal_value = env.optimal_value
        optimal_arm = env.optimal_arm

        arm = agent.select_action()
        reward = env.pull(arm)
        agent.update(arm, reward)

        rewards[t] = reward
        chosen_optimal[t] = arm == optimal_arm
        instant_regret[t] = optimal_value - env.expected_value(arm)

    return {
        "rewards": rewards,
        "chosen_optimal": chosen_optimal,
        "cumulative_regret": np.cumsum(instant_regret),
        "optimal_action_rate": np.cumsum(chosen_optimal) / (np.arange(steps) + 1),
    }
