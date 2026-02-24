"""Tests for LoggedEnv and offline experiment runner."""

import numpy as np
import pytest

from mab.environments.replay import LoggedEnv
from mab.bandits import SWUCB
from mab.experiment import run_offline_trial, run_offline_experiment


def _make_logged_env(k: int = 3, T: int = 100, seed: int = 0) -> LoggedEnv:
    rng = np.random.default_rng(seed)
    logged_arms = rng.integers(0, k, size=T)
    logged_rewards = rng.integers(0, 2, size=T).astype(float)
    return LoggedEnv(n_arms=k, logged_arms=logged_arms, logged_rewards=logged_rewards)


class TestLoggedEnv:
    def test_n_arms_and_T(self):
        env = _make_logged_env(k=4, T=50)
        assert env.n_arms == 4
        assert env.T == 50

    def test_reset_is_noop(self):
        """reset() should not raise and env should remain functional."""
        env = _make_logged_env()
        env.reset()
        reward = env.query(0, int(env.logged_arms[0]))
        assert reward is not None

    def test_query_match_returns_reward(self):
        arms = np.array([1, 0, 2])
        rewards = np.array([0.0, 1.0, 1.0])
        env = LoggedEnv(n_arms=3, logged_arms=arms, logged_rewards=rewards)
        assert env.query(0, 1) == pytest.approx(0.0)
        assert env.query(1, 0) == pytest.approx(1.0)
        assert env.query(2, 2) == pytest.approx(1.0)

    def test_query_mismatch_returns_none(self):
        arms = np.array([1, 0, 2])
        rewards = np.array([0.0, 1.0, 1.0])
        env = LoggedEnv(n_arms=3, logged_arms=arms, logged_rewards=rewards)
        assert env.query(0, 0) is None
        assert env.query(0, 2) is None
        assert env.query(1, 1) is None

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            LoggedEnv(
                n_arms=2,
                logged_arms=np.array([0, 1]),
                logged_rewards=np.array([1.0]),
            )

    def test_logged_arms_property(self):
        arms = np.array([2, 0, 1])
        env = LoggedEnv(n_arms=3, logged_arms=arms, logged_rewards=np.zeros(3))
        np.testing.assert_array_equal(env.logged_arms, arms)


class TestOfflineExperiment:
    def _bandit_factory(self, k: int = 3):
        return lambda: SWUCB(k=k, eps=1.0, M=20, alpha=0.1)

    def test_run_offline_trial_returns_nonneg_reward(self):
        env = _make_logged_env(k=3, T=200)
        reward, valid = run_offline_trial(
            bandit_factory=self._bandit_factory(3),
            env_factory=lambda: env,
        )
        assert reward >= 0.0
        assert 0 <= valid <= 200

    def test_valid_events_at_most_T(self):
        env = _make_logged_env(k=3, T=500)
        _, valid = run_offline_trial(
            bandit_factory=self._bandit_factory(3),
            env_factory=lambda: env,
        )
        assert valid <= env.T

    def test_run_offline_experiment_shapes(self):
        env = _make_logged_env(k=3, T=100)
        rewards, valid = run_offline_experiment(
            bandit_factory=self._bandit_factory(3),
            env_factory=lambda: env,
            n_trials=5,
            n_jobs=1,
        )
        assert rewards.shape == (5,)
        assert valid.shape == (5,)
        assert np.all(rewards >= 0)
        assert np.all(valid >= 0)

    def test_ctr_bounded(self):
        """CTR must be in [0, 1] since rewards are binary."""
        env = _make_logged_env(k=3, T=300)
        rewards, valid = run_offline_experiment(
            bandit_factory=self._bandit_factory(3),
            env_factory=lambda: env,
            n_trials=5,
            n_jobs=1,
        )
        ctr = rewards / np.maximum(valid, 1)
        assert np.all(ctr >= 0.0)
        assert np.all(ctr <= 1.0)

    def test_deterministic_bandit_gives_fixed_valid_count(self):
        """With alpha=0 and deterministic UCB the valid count is deterministic."""
        arms = np.array([0, 1, 2, 0, 1, 2] * 10)
        rews = np.ones(60, dtype=float)
        env = LoggedEnv(n_arms=3, logged_arms=arms, logged_rewards=rews)

        # A bandit that always picks arm 0.
        class AlwaysZero:
            def select_arm(self): return 0
            def update(self, arm, reward): pass
            def reset(self): pass

        reward, valid = run_offline_trial(
            bandit_factory=AlwaysZero,
            env_factory=lambda: env,
        )
        # arm 0 appears at indices 0, 3, 6, … (every 3rd event) → 20 valid events
        assert valid == 20
        assert reward == pytest.approx(20.0)
