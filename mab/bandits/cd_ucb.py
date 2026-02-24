"""CD-UCB: Change-Detection Upper Confidence Bound.

A standard UCB algorithm where each arm has an independent change detector.
When the detector for a played arm fires, that arm's history is reset so the
UCB index reflects only post-change observations.

Arm selection (Liu et al. 2018, Algorithm 1):
  - With probability alpha: play a uniformly random arm (forced exploration).
  - Otherwise: play arm i* = argmax_i [ avg(i) + sqrt(xi * log(n_t) / N_t(i)) ]
    where n_t = sum of active counts across all arms, N_t(i) = plays of arm i
    since its last reset.
    xi = 1 is required for the regret bound (Theorems 1 and 3).

On alarm for arm i: reset N_t(i) = 0, avg(i) = 0, and the detector state.

warmup (Algorithm 3, CUSUM-UCB only):
  Each arm starts with a countdown of `warmup` forced plays so the detector can
  estimate the pre-change mean û₀ before normal UCB selection begins.  After an
  alarm on arm i the countdown is restarted at `warmup`.  Set warmup=0 (default)
  for PHT-UCB and other detectors that need no burn-in.
"""

from dataclasses import dataclass, field
from typing import Type

import numpy as np

from .base import BanditAlgorithm
from ..detectors.base import ChangeDetector


@dataclass
class CDUCB(BanditAlgorithm):
    k: int                              # number of arms
    xi: float                           # UCB exploration coefficient (ξ in paper; use 1.0)
    alpha: float                        # forced-exploration probability
    detector_cls: Type[ChangeDetector]  # which CD algorithm to use
    detector_kwargs: dict = field(default_factory=dict)
    warmup: int = 0                     # forced plays per arm on init/reset (M for CUSUM-UCB)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.counts = np.zeros(self.k)
        self.avgs = np.zeros(self.k)
        self._cnt = np.full(self.k, self.warmup, dtype=int)
        self._detectors: list[ChangeDetector] = [
            self.detector_cls(**self.detector_kwargs) for _ in range(self.k)
        ]

    def select_arm(self) -> int:
        # Algorithm 3: serve any arm still in its warmup countdown first.
        pending = np.where(self._cnt > 0)[0]
        if len(pending) > 0:
            arm = int(pending[0])
            self._cnt[arm] -= 1
            return arm

        if np.random.random() < self.alpha:
            return int(np.random.randint(self.k))

        n_t = int(np.sum(self.counts))
        log_nt = np.log(max(1, n_t))

        # Arms never played get infinite bonus (must be explored first).
        safe_counts = np.where(self.counts == 0, 1.0, self.counts)
        ucb = np.where(
            self.counts == 0,
            np.inf,
            self.avgs + np.sqrt(self.xi * log_nt / safe_counts),
        )
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        # Incremental mean update: avg_new = avg + (x - avg) / n
        self.avgs[arm] += (reward - self.avgs[arm]) / self.counts[arm]

        # Run change detector for this arm.
        if self._detectors[arm].update(reward):
            self.counts[arm] = 0
            self.avgs[arm] = 0.0
            self._cnt[arm] = self.warmup
            self._detectors[arm].reset()
