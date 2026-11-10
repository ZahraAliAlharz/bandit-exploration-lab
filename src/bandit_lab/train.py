"""CLI entry point for a single bandit run."""
from __future__ import annotations

import argparse

import numpy as np

from bandit_lab.agents import EpsilonGreedy, GradientBandit, ThompsonSamplingBeta, UCB1
from bandit_lab.envs import BernoulliBandit
from bandit_lab.metrics import run_bandit

AGENTS = {
    "epsilon-greedy": lambda k, seed: EpsilonGreedy(k, epsilon=0.1, seed=seed),
    "ucb1": lambda k, seed: UCB1(k, c=2.0),
    "thompson": lambda k, seed: ThompsonSamplingBeta(k, seed=seed),
    "gradient": lambda k, seed: GradientBandit(k, seed=seed),
}


def main():
    parser = argparse.ArgumentParser(description="Run one bandit agent against one bandit problem.")
    parser.add_argument("agent", choices=AGENTS.keys())
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = BernoulliBandit(args.k, seed=args.seed)
    agent = AGENTS[args.agent](args.k, args.seed)
    result = run_bandit(env, agent, args.steps)

    print(f"Arm probabilities: {np.round(env.probs, 3)}")
    print(f"Optimal arm: {env.optimal_arm} (p={env.optimal_value:.3f})")
    print(f"Final cumulative regret: {result['cumulative_regret'][-1]:.2f}")
    print(f"Final optimal-action rate: {result['optimal_action_rate'][-1]:.3f}")


if __name__ == "__main__":
    main()
