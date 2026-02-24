"""Switching environment with random mean changes per arm."""

from dataclasses import dataclass

import numpy as np

from .base import Environment


@dataclass
class SwitchingEnv(Environment):
    _T: int     # horizon
    k: int      # number of arms
    beta: float  # per-step switch probability (hazard rate)

    @property
    def n_arms(self) -> int:
        return self.k

    @property
    def T(self) -> int:
        return self._T

    def reset(self) -> None:
        self._mus = np.random.uniform(0.0, 1.0, self.k)

    def step(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        # Each arm switches independently with probability beta.
        switches = np.random.random(self.k) < self.beta
        new_means = np.random.uniform(0.0, 1.0, self.k)
        self._mus = np.where(switches, new_means, self._mus)

        rewards = np.random.binomial(1, self._mus).astype(float)
        return rewards, self._mus.copy()
