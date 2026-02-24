"""Switching environment (Mellor & Shapiro 2013; Liu et al. 2018, Section V-A).

Each arm independently switches its mean at each timestep with probability beta
(the hazard rate).  When a switch occurs the new mean is drawn from U[0, 1].

  mu_t(i) = mu_{t-1}(i)      with probability 1 - beta
           = U[0, 1] sample  with probability beta

Initial means mu_0(i) ~ U[0, 1] for all i.  All rewards are Bernoulli.

Using a constant hazard function beta = gamma_T / T (as in the paper's
experiments) gives an expected total of gamma_T breakpoints across all arms
over the horizon T.
"""

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
