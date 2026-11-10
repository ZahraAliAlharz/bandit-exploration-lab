"""Compare exploration strategies on a stationary 10-armed Bernoulli bandit.

Generates results/stationary_regret.png and results/stationary_optimal_action.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bandit_lab.agents import EpsilonGreedy, GradientBandit, ThompsonSamplingBeta, UCB1
from bandit_lab.envs import BernoulliBandit
from bandit_lab.metrics import run_bandit

K = 10
STEPS = 2000
SEEDS = range(30)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

AGENT_FACTORIES = {
    "Epsilon-greedy (eps=0.1)": lambda seed: EpsilonGreedy(K, epsilon=0.1, seed=seed),
    "UCB1 (c=2)": lambda seed: UCB1(K, c=2.0),
    "Thompson sampling": lambda seed: ThompsonSamplingBeta(K, seed=seed),
    "Gradient bandit": lambda seed: GradientBandit(K, step_size=0.1, seed=seed),
}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    regret_curves = {name: [] for name in AGENT_FACTORIES}
    optimal_curves = {name: [] for name in AGENT_FACTORIES}
    final_regret = {name: [] for name in AGENT_FACTORIES}

    for seed in SEEDS:
        true_probs = np.random.default_rng(seed).uniform(0.0, 1.0, size=K)
        for name, factory in AGENT_FACTORIES.items():
            env = BernoulliBandit(K, probs=true_probs, seed=seed + 10_000)
            agent = factory(seed)
            result = run_bandit(env, agent, STEPS)
            regret_curves[name].append(result["cumulative_regret"])
            optimal_curves[name].append(result["optimal_action_rate"])
            final_regret[name].append(result["cumulative_regret"][-1])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curves in regret_curves.items():
        mean_curve = np.mean(curves, axis=0)
        ax.plot(mean_curve, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative regret")
    ax.set_title(f"Cumulative regret, {K}-armed Bernoulli bandit (mean over {len(SEEDS)} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stationary_regret.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curves in optimal_curves.items():
        mean_curve = np.mean(curves, axis=0)
        ax.plot(mean_curve, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("% optimal action")
    ax.set_title(f"Optimal-action rate, {K}-armed Bernoulli bandit (mean over {len(SEEDS)} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stationary_optimal_action.png", dpi=150)

    print(f"{'Agent':<28} {'Final cumulative regret (mean ± std)':<40}")
    for name, vals in final_regret.items():
        print(f"{name:<28} {np.mean(vals):6.2f} ± {np.std(vals):5.2f}")


if __name__ == "__main__":
    main()
