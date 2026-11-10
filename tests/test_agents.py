import numpy as np

from bandit_lab.agents import EpsilonGreedy, GradientBandit, ThompsonSamplingBeta, UCB1


def test_epsilon_greedy_action_in_range():
    agent = EpsilonGreedy(k=5, epsilon=0.1, seed=0)
    for _ in range(50):
        arm = agent.select_action()
        assert 0 <= arm < 5
        agent.update(arm, reward=1.0)


def test_epsilon_greedy_zero_epsilon_is_greedy():
    agent = EpsilonGreedy(k=3, epsilon=0.0, seed=0)
    agent.q = np.array([0.1, 0.9, 0.2])
    assert agent.select_action() == 1


def test_epsilon_greedy_sample_average_converges_to_true_mean():
    agent = EpsilonGreedy(k=1, epsilon=0.0, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(2000):
        agent.update(0, rng.normal(3.0, 1.0))
    assert abs(agent.q[0] - 3.0) < 0.2


def test_epsilon_greedy_constant_step_tracks_recent_rewards_more():
    agent = EpsilonGreedy(k=1, epsilon=0.0, step_size=0.1, seed=0)
    for _ in range(100):
        agent.update(0, 0.0)
    for _ in range(20):
        agent.update(0, 1.0)
    # A constant step size should have moved noticeably toward the new
    # reward level, unlike a sample average which would barely budge
    # after 100 prior updates.
    assert agent.q[0] > 0.3


def test_ucb1_plays_each_arm_at_least_once_first():
    agent = UCB1(k=4, c=2.0)
    seen = set()
    for _ in range(4):
        arm = agent.select_action()
        seen.add(arm)
        agent.update(arm, reward=0.5)
    assert seen == {0, 1, 2, 3}


def test_thompson_sampling_converges_to_best_arm_fraction():
    agent = ThompsonSamplingBeta(k=3, seed=0)
    true_probs = [0.1, 0.8, 0.3]
    rng = np.random.default_rng(0)
    choices = []
    for _ in range(3000):
        arm = agent.select_action()
        reward = float(rng.random() < true_probs[arm])
        agent.update(arm, reward)
        choices.append(arm)
    late_choices = choices[-200:]
    assert late_choices.count(1) / len(late_choices) > 0.8


def test_gradient_bandit_policy_sums_to_one():
    agent = GradientBandit(k=5, seed=0)
    probs = agent._policy()
    assert np.isclose(probs.sum(), 1.0)


def test_gradient_bandit_shifts_preference_toward_rewarding_arm():
    agent = GradientBandit(k=2, step_size=0.5, seed=0)
    for _ in range(100):
        agent.update(0, reward=1.0)
        agent.update(1, reward=0.0)
    assert agent.preferences[0] > agent.preferences[1]
