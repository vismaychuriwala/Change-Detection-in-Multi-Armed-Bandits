"""Tests for change detection algorithms."""

import numpy as np
import pytest

from mab.detectors import CUSUM, PHT


class TestCUSUM:
    def _make(self, eps=0.05, M=20, h=3.0):
        return CUSUM(eps=eps, M=M, h=h)

    def test_no_alarm_during_burnin(self):
        """CUSUM must not alarm while collecting the first M samples."""
        det = self._make(M=20)
        rng = np.random.default_rng(0)
        for _ in range(20):
            alarm = det.update(float(rng.binomial(1, 0.5)))
            assert not alarm

    def test_no_alarm_on_stationary_data(self):
        """CUSUM should rarely alarm on stationary Bernoulli(0.5) data.

        After each alarm the detector is reset (as the bandit would do),
        so we count distinct false-change-point detections, not repeated
        firings from walks that were never cleared.
        """
        det = self._make(eps=0.05, M=50, h=8.0)
        rng = np.random.default_rng(1)
        alarm_count = 0
        for _ in range(500):
            if det.update(float(rng.binomial(1, 0.5))):
                alarm_count += 1
                det.reset()  # mimic bandit behaviour
        assert alarm_count <= 2

    def test_detects_mean_shift(self):
        """CUSUM should detect a large mean shift well within 200 samples."""
        det = self._make(eps=0.05, M=50, h=3.0)
        rng = np.random.default_rng(2)
        # Burn-in at mu=0.5
        for _ in range(50):
            det.update(float(rng.binomial(1, 0.5)))
        # Post-change at mu=0.9
        detected = False
        for _ in range(200):
            if det.update(float(rng.binomial(1, 0.9))):
                detected = True
                break
        assert detected

    def test_reset_clears_state(self):
        """After reset, CUSUM should behave as if freshly initialised."""
        det = self._make(M=5, h=0.1)
        rng = np.random.default_rng(3)
        # Drive the detector to near-alarm state.
        for _ in range(5):
            det.update(float(rng.binomial(1, 0.5)))
        det.reset()
        # After reset, burn-in counter should be 0 — no alarm for first M samples.
        for _ in range(5):
            alarm = det.update(float(rng.binomial(1, 0.5)))
            assert not alarm


class TestPHT:
    def _make(self, eps=0.05, h=3.0):
        return PHT(eps=eps, h=h)

    def test_no_alarm_on_stationary_data(self):
        """PHT should rarely alarm on stationary Bernoulli(0.5) data.

        Detector is reset after each alarm to count distinct false detections.
        """
        det = self._make(eps=0.05, h=8.0)
        rng = np.random.default_rng(10)
        alarm_count = 0
        for _ in range(500):
            if det.update(float(rng.binomial(1, 0.5))):
                alarm_count += 1
                det.reset()
        assert alarm_count <= 2

    def test_detects_mean_shift(self):
        """PHT should detect a large mean shift."""
        det = self._make(eps=0.05, h=3.0)
        rng = np.random.default_rng(11)
        # First 100 samples at mu=0.5.
        for _ in range(100):
            det.update(float(rng.binomial(1, 0.5)))
        # Post-change at mu=0.9.
        detected = False
        for _ in range(300):
            if det.update(float(rng.binomial(1, 0.9))):
                detected = True
                break
        assert detected

    def test_reset_clears_state(self):
        det = self._make()
        rng = np.random.default_rng(12)
        for _ in range(50):
            det.update(float(rng.binomial(1, 0.5)))
        det.reset()
        # Internal sum and count should be zero again.
        assert det._sum == 0.0
        assert det._n == 0
        assert det._g_pos == 0.0
        assert det._g_neg == 0.0
