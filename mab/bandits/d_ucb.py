"""D-UCB: Discounted Upper Confidence Bound (Kocsis & Szepesvári 2006).

Another passively adaptive baseline. Older rewards are down-weighted
geometrically by factor gamma in (0, 1), so the algorithm naturally forgets
stale observations without explicit change detection.

Discounted statistics updated O(K) per step by decaying running sums:
  n_gamma[i] = sum_{s=1}^{t} gamma^{t-s} * 1_{I_s = i}
  r_gamma[i] = sum_{s=1}^{t} gamma^{t-s} * X_s(i) * 1_{I_s = i}
  avg_gamma[i] = r_gamma[i] / n_gamma[i]

UCB index: avg_gamma(i) + B * sqrt(eps * log(n_t) / n_gamma(i))
  where n_t = sum_i n_gamma(i)  (total discounted count)
        B   = reward range (1 for Bernoulli)
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
        # Decay all arms before adding new observation.
        self._n_gamma *= self.gamma
        self._r_gamma *= self.gamma
        # Update played arm.
        self._n_gamma[arm] += 1.0
        self._r_gamma[arm] += reward
