"""Page-Hinkley Test (PHT) change detector.

PHT is a variant of two-sided CUSUM where the baseline u0 is the running
mean of all samples seen so far (rather than the mean of the first M samples).
This makes PHT more adaptive to the pre-change distribution.

Algorithm:
  - At each step k, u0_k = (1/k) * sum_{i=1}^{k} y_i  (running mean)
  - s+_k = y_k - u0_k - eps
  - s-_k = u0_k - y_k - eps
  - g+_k = max(0, g+_{k-1} + s+_k)
  - g-_k = max(0, g-_{k-1} + s-_k)
  - Alarm if g+_k >= h or g-_k >= h.
"""

from dataclasses import dataclass

from .base import ChangeDetector


@dataclass
class PHT(ChangeDetector):
    eps: float
    h: float

    def __post_init__(self) -> None:
        self._sum: float = 0.0
        self._n: int = 0
        self._g_pos: float = 0.0
        self._g_neg: float = 0.0

    def update(self, reward: float) -> bool:
        self._n += 1
        self._sum += reward
        u0 = self._sum / self._n  # running mean

        s_pos = reward - u0 - self.eps
        s_neg = u0 - reward - self.eps
        self._g_pos = max(0.0, self._g_pos + s_pos)
        self._g_neg = max(0.0, self._g_neg + s_neg)

        return self._g_pos >= self.h or self._g_neg >= self.h

    def reset(self) -> None:
        self._sum = 0.0
        self._n = 0
        self._g_pos = 0.0
        self._g_neg = 0.0
