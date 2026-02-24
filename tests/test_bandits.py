"""Tests for bandit algorithms."""

import numpy as np
import pytest

from mab.bandits import CDUCB, SWUCB, DUCB
from mab.detectors import CUSUM, PHT


def _run_bandit(bandit, T: int = 200, k: int = 2, seed: int = 0) -> float:
    """Run a bandit on random Bernoulli rewards and return total regret."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0, 1, k)
    total_regret = 0.0
    for _ in range(T):
        arm = bandit.select_arm()
        reward = float(rng.binomial(1, mu[arm]))
        bandit.update(arm, reward)
        total_regret += float(np.max(mu) - mu[arm])
    return total_regret


class TestCDUCB:
    def _make_cusum(self, k=2):
        return CDUCB(
            k=k, xi=1.0, alpha=0.1,
            detector_cls=CUSUM,
            detector_kwargs={"eps": 0.1, "M": 20, "h": 3.0},
            warmup=20,
        )

    def _make_pht(self, k=2):
        return CDUCB(
            k=k, xi=1.0, alpha=0.1,
            detector_cls=PHT,
            detector_kwargs={"eps": 0.1, "h": 3.0},
        )

    def test_cusum_runs_without_error(self):
        bandit = self._make_cusum()
        regret = _run_bandit(bandit, T=300)
        assert regret >= 0.0

    def test_pht_runs_without_error(self):
        bandit = self._make_pht()
        regret = _run_bandit(bandit, T=300)
        assert regret >= 0.0

    def test_select_arm_in_range(self):
        k = 5
        bandit = self._make_cusum(k=k)
        rng = np.random.default_rng(42)
        for _ in range(50):
            arm = bandit.select_arm()
            assert 0 <= arm < k
            bandit.update(arm, float(rng.random()))

    def test_reset_clears_state(self):
        bandit = self._make_cusum()
        _run_bandit(bandit, T=100)
        bandit.reset()
        assert np.all(bandit.counts == 0)
        assert np.all(bandit.avgs == 0)

    def test_unplayed_arms_selected_first(self):
        """With alpha=0 and no warmup, unplayed arms get infinite bonus and are chosen first."""
        bandit = CDUCB(
            k=3, xi=1.0, alpha=0.0,
            detector_cls=CUSUM,
            detector_kwargs={"eps": 0.1, "M": 5, "h": 10.0},
        )
        arms_played = set()
        rng = np.random.default_rng(99)
        for _ in range(3):
            arm = bandit.select_arm()
            arms_played.add(arm)
            bandit.update(arm, float(rng.random()))
        assert arms_played == {0, 1, 2}


class TestSWUCB:
    def _make(self, k=2):
        return SWUCB(k=k, eps=1.0, M=30, alpha=0.1)

    def test_runs_without_error(self):
        bandit = self._make()
        regret = _run_bandit(bandit, T=300)
        assert regret >= 0.0

    def test_window_size_respected(self):
        """Window for each arm should never exceed M."""
        bandit = SWUCB(k=2, eps=1.0, M=10, alpha=0.0)
        rng = np.random.default_rng(7)
        for _ in range(50):
            arm = bandit.select_arm()
            bandit.update(arm, float(rng.random()))
        for w in bandit._windows:
            assert len(w) <= 10

    def test_reset_clears_windows(self):
        bandit = self._make()
        _run_bandit(bandit, T=100)
        bandit.reset()
        for w in bandit._windows:
            assert len(w) == 0


class TestDUCB:
    def _make(self, k=2):
        return DUCB(k=k, gamma=0.9, eps=1.0)

    def test_runs_without_error(self):
        bandit = self._make()
        regret = _run_bandit(bandit, T=300)
        assert regret >= 0.0

    def test_discount_decays_old_observations(self):
        """After many steps, early observations should have negligible weight."""
        bandit = DUCB(k=1, gamma=0.5, eps=1.0)
        # Play arm 0 many times.
        for _ in range(50):
            bandit.update(0, 1.0)
        # Effective count should be bounded: sum_{t=0}^{49} 0.5^t < 2.
        assert bandit._n_gamma[0] < 2.1

    def test_reset_clears_state(self):
        bandit = self._make()
        _run_bandit(bandit, T=100)
        bandit.reset()
        assert np.all(bandit._n_gamma == 0)
        assert np.all(bandit._r_gamma == 0)
