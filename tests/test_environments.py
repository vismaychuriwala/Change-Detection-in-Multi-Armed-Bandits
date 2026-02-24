"""Tests for MAB environments."""

import numpy as np
import pytest

from mab.environments import FlippingEnv, SwitchingEnv


class TestFlippingEnv:
    def _make(self, T=999, delta=0.1):
        env = FlippingEnv(_T=T, delta=delta)
        env.reset()
        return env

    def test_n_arms_and_T(self):
        env = self._make()
        assert env.n_arms == 2
        assert env.T == 999

    def test_arm0_always_half(self):
        """Arm 0 should always have mean 0.5."""
        env = self._make(T=300)
        for t in range(300):
            _, means = env.step(t)
            assert means[0] == pytest.approx(0.5)

    def test_arm1_phase_means(self):
        """Arm 1 should have the correct mean in each phase."""
        T, delta = 300, 0.1
        env = self._make(T=T, delta=delta)
        t1, t2 = T // 3, 2 * T // 3

        _, m_early = env.step(0)
        assert m_early[1] == pytest.approx(0.5 - delta)

        _, m_mid = env.step(t1)
        assert m_mid[1] == pytest.approx(0.8)

        _, m_late = env.step(t2)
        assert m_late[1] == pytest.approx(0.5 - delta)

    def test_rewards_are_binary(self):
        """Sampled rewards must be 0 or 1 (Bernoulli)."""
        env = self._make(T=100)
        for t in range(100):
            rewards, _ = env.step(t)
            assert set(rewards).issubset({0.0, 1.0})

    def test_reset_reinitialises(self):
        """reset() should produce a fresh mu matrix (deterministic for this env)."""
        env = FlippingEnv(_T=100, delta=0.2)
        env.reset()
        _, m1 = env.step(0)
        env.reset()
        _, m2 = env.step(0)
        np.testing.assert_array_equal(m1, m2)


class TestSwitchingEnv:
    def _make(self, T=500, k=5, beta=0.1):
        env = SwitchingEnv(_T=T, k=k, beta=beta)
        env.reset()
        return env

    def test_n_arms_and_T(self):
        env = self._make()
        assert env.n_arms == 5
        assert env.T == 500

    def test_means_in_unit_interval(self):
        env = self._make()
        for t in range(100):
            _, means = env.step(t)
            assert np.all(means >= 0.0) and np.all(means <= 1.0)

    def test_rewards_are_binary(self):
        env = self._make()
        for t in range(50):
            rewards, _ = env.step(t)
            assert set(rewards).issubset({0.0, 1.0})

    def test_switch_rate_approximately_correct(self):
        """Over many steps, each arm should switch ~beta fraction of steps."""
        T, k, beta = 10_000, 5, 0.2
        env = SwitchingEnv(_T=T, k=k, beta=beta)
        env.reset()

        prev_means = np.full(k, -1.0)
        switches = 0
        for t in range(T):
            _, means = env.step(t)
            if t > 0:
                switches += int(np.any(means != prev_means))
            prev_means = means.copy()

        # Approximate: at least one arm switches in ~1-(1-beta)^k fraction of steps.
        expected_fraction = 1.0 - (1.0 - beta) ** k
        actual_fraction = switches / (T - 1)
        assert abs(actual_fraction - expected_fraction) < 0.05

    def test_reset_draws_new_initial_means(self):
        """Two resets should (almost certainly) produce different initial means."""
        env = SwitchingEnv(_T=10, k=5, beta=0.0)
        np.random.seed(0)
        env.reset()
        _, m1 = env.step(0)
        np.random.seed(1)
        env.reset()
        _, m2 = env.step(0)
        assert not np.allclose(m1, m2)
