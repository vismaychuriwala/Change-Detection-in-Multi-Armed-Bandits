"""Page-Hinkley Test (PHT) change detector.

Uses a running mean baseline.
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
        u0 = self._sum / self._n

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
