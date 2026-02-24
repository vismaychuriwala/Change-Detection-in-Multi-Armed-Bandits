"""Flipping environment (Liu et al. 2018, Section V-A).

Two arms with Bernoulli rewards:
  Arm 0: stationary at mu = 0.5 throughout.
  Arm 1: mu = 0.5 - delta  for  t in [0, T/3)  and  [2T/3, T)
          mu = 0.8          for  t in [T/3, 2T/3)

Two change points occur at T/3 and 2T/3.  The parameter delta controls how
large the gap is between phases; larger delta makes changes easier to detect.

The mu matrix is precomputed on reset() and rewards are drawn Bernoulli(mu).
"""

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
