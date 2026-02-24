"""SW-UCB: Sliding Window Upper Confidence Bound.

Maintains a fixed-size window of recent rewards per arm for adaptation.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .base import BanditAlgorithm


@dataclass
class SWUCB(BanditAlgorithm):
    k: int      # number of arms
    eps: float  # UCB exploration coefficient
    M: int      # sliding window size
    alpha: float  # forced-exploration probability

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._windows: list[deque[float]] = [deque(maxlen=self.M) for _ in range(self.k)]

    def select_arm(self) -> int:
        if np.random.random() < self.alpha:
            return int(np.random.randint(self.k))

        sizes = np.array([len(w) for w in self._windows])
        n_t = int(sizes.sum())
        log_nt = np.log(max(1, n_t))

        ucb = np.where(
            sizes == 0,
            np.inf,
            np.array([np.mean(w) if w else 0.0 for w in self._windows])
            + np.sqrt(self.eps * log_nt / np.maximum(sizes, 1)),
        )
        ucb[sizes == 0] = np.inf
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self._windows[arm].append(reward)
