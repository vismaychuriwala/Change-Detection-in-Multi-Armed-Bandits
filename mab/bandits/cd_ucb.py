"""CD-UCB: Change-Detection Upper Confidence Bound.

Combines UCB with per-arm change detection to reset arm histories on detected shifts.
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
        pending = np.where(self._cnt > 0)[0]
        if len(pending) > 0:
            arm = int(pending[0])
            self._cnt[arm] -= 1
            return arm

        if np.random.random() < self.alpha:
            return int(np.random.randint(self.k))

        n_t = int(np.sum(self.counts))
        log_nt = np.log(max(1, n_t))

        safe_counts = np.where(self.counts == 0, 1.0, self.counts)
        ucb = np.where(
            self.counts == 0,
            np.inf,
            self.avgs + np.sqrt(self.xi * log_nt / safe_counts),
        )
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.avgs[arm] += (reward - self.avgs[arm]) / self.counts[arm]

        if self._detectors[arm].update(reward):
            self.counts[arm] = 0
            self.avgs[arm] = 0.0
            self._cnt[arm] = self.warmup
            self._detectors[arm].reset()
