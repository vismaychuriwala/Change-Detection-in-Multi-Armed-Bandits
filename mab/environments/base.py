from abc import ABC, abstractmethod

import numpy as np


class Environment(ABC):
    """Abstract base class for MAB environments.

    At each timestep t the environment returns sampled rewards for all arms
    AND the true expected rewards (means) used by the experiment runner to
    compute regret. Only the reward for the played arm is actually observed
    by the bandit; the means are never exposed to the bandit algorithm.
    """

    @property
    @abstractmethod
    def n_arms(self) -> int:
        """Number of arms."""

    @property
    @abstractmethod
    def T(self) -> int:
        """Horizon length."""

    @abstractmethod
    def reset(self) -> None:
        """Re-initialise the environment (called at the start of each trial)."""

    @abstractmethod
    def step(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (rewards, means) for all arms at timestep t.

        rewards: shape (n_arms,) — sampled reward for each arm.
        means:   shape (n_arms,) — true expected reward mu_t(i) for each arm.
        """
