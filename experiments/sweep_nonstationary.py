"""Sample-average vs constant step-size epsilon-greedy on a drifting bandit.

Reproduces the qualitative result of Sutton & Barto (2018), exercise 2.5:
in a non-stationary environment, weighting recent rewards more heavily
(constant step size) tracks the moving optimum better than an unweighted
sample average, whose step size (1/n) shrinks to near zero and stops
adapting. Generates results/nonstationary_comparison.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bandit_lab.agents import EpsilonGreedy
from bandit_lab.envs import DriftingBernoulliBandit
from bandit_lab.metrics import run_bandit

K = 10
STEPS = 10_000
SEEDS = range(30)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

AGENT_FACTORIES = {
    "Sample average (eps=0.1)": lambda seed: EpsilonGreedy(K, epsilon=0.1, seed=seed),
    "Constant step size, alpha=0.1 (eps=0.1)": lambda seed: EpsilonGreedy(K, epsilon=0.1, step_size=0.1, seed=seed),
}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    optimal_curves = {name: [] for name in AGENT_FACTORIES}
    final_rate = {name: [] for name in AGENT_FACTORIES}

    for seed in SEEDS:
        for name, factory in AGENT_FACTORIES.items():
            env = DriftingBernoulliBandit(K, probs=[0.5] * K, drift_std=0.02, seed=seed)
            agent = factory(seed)
            result = run_bandit(env, agent, STEPS)
            optimal_curves[name].append(result["optimal_action_rate"])
            final_rate[name].append(result["optimal_action_rate"][-1])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curves in optimal_curves.items():
        mean_curve = np.mean(curves, axis=0)
        ax.plot(mean_curve, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("% optimal action")
    ax.set_title(f"Drifting bandit: sample average vs constant step size (mean over {len(SEEDS)} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "nonstationary_comparison.png", dpi=150)

    print(f"{'Agent':<42} {'Final optimal-action rate (mean ± std)':<40}")
    for name, vals in final_rate.items():
        print(f"{name:<42} {np.mean(vals):.3f} ± {np.std(vals):.3f}")


if __name__ == "__main__":
    main()
