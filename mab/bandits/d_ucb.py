"""D-UCB: Discounted Upper Confidence Bound (Kocsis & Szepesvári 2006).

Uses geometric discounting to forget old rewards in non-stationary settings.
"""

from dataclasses import dataclass

import numpy as np

from .base import BanditAlgorithm


@dataclass
class DUCB(BanditAlgorithm):
    k: int          # number of arms
    gamma: float    # discount factor in (0, 1)
    eps: float      # UCB exploration coefficient
    B: float = 1.0  # reward range (1 for [0,1]-valued rewards)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._n_gamma = np.zeros(self.k)   # discounted play counts
        self._r_gamma = np.zeros(self.k)   # discounted reward sums

    def select_arm(self) -> int:
        n_t = self._n_gamma.sum()
        log_nt = np.log(max(1.0, n_t))

        safe_n = np.where(self._n_gamma == 0, 1.0, self._n_gamma)
        ucb = np.where(
            self._n_gamma == 0,
            np.inf,
            self._r_gamma / safe_n
            + self.B * np.sqrt(self.eps * log_nt / safe_n),
        )
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self._n_gamma *= self.gamma
        self._r_gamma *= self.gamma
        self._n_gamma[arm] += 1.0
        self._r_gamma[arm] += reward
