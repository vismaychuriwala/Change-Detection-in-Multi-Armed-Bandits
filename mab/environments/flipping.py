"""Flipping environment with two arms and scheduled mean changes."""

from dataclasses import dataclass

import numpy as np

from .base import Environment


@dataclass
class FlippingEnv(Environment):
    _T: int       # horizon
    delta: float  # mean gap parameter Δ
    k: int = 2    # number of arms (kept as a parameter for generality)

    @property
    def n_arms(self) -> int:
        return self.k

    @property
    def T(self) -> int:
        return self._T

    def reset(self) -> None:
        T, k, delta = self._T, self.k, self.delta
        self._mus = np.full((T, k), 0.5)

        # Arm 1 flips: bad in first and last third, good in middle.
        t1 = T // 3
        t2 = 2 * T // 3
        self._mus[:t1, 1] = 0.5 - delta
        self._mus[t1:t2, 1] = 0.8
        self._mus[t2:, 1] = 0.5 - delta

    def step(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        means = self._mus[t]
        rewards = np.random.binomial(1, means).astype(float)
        return rewards, means
