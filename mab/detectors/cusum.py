"""Two-sided CUSUM change detector.

Algorithm (Liu et al. 2018, Algorithm 2):
  - Collect first M samples to estimate the pre-change mean u0.
  - For each subsequent sample y_k:
      s+_k = y_k - u0 - eps
      s-_k = u0 - y_k - eps
      g+_k = max(0, g+_{k-1} + s+_k)
      g-_k = max(0, g-_{k-1} + s-_k)
  - Alarm if g+_k >= h or g-_k >= h.

Key difference from PHT: the baseline u0 is fixed using the first M samples.
"""

from dataclasses import dataclass, field

import numpy as np

from .base import ChangeDetector


@dataclass
class CUSUM(ChangeDetector):
    eps: float
    M: int
    h: float

    def __post_init__(self) -> None:
        self._buffer: list[float] = []
        self._u0: float = 0.0
        self._g_pos: float = 0.0
        self._g_neg: float = 0.0
        self._n: int = 0

    def update(self, reward: float) -> bool:
        self._n += 1

        # Burn-in phase: accumulate M samples to fix baseline u0.
        if self._n <= self.M:
            self._buffer.append(reward)
            if self._n == self.M:
                self._u0 = float(np.mean(self._buffer))
            return False

        # Detection phase: update walks using fixed u0.
        s_pos = reward - self._u0 - self.eps
        s_neg = self._u0 - reward - self.eps
        self._g_pos = max(0.0, self._g_pos + s_pos)
        self._g_neg = max(0.0, self._g_neg + s_neg)

        return self._g_pos >= self.h or self._g_neg >= self.h

    def reset(self) -> None:
        self._buffer = []
        self._u0 = 0.0
        self._g_pos = 0.0
        self._g_neg = 0.0
        self._n = 0
