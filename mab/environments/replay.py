"""Offline bandit environment using logged (arm, reward) data."""

import numpy as np


class LoggedEnv:
    """Offline bandit environment backed by logged data."""

    def __init__(
        self,
        n_arms: int,
        logged_arms: np.ndarray,
        logged_rewards: np.ndarray,
    ) -> None:
        if logged_arms.shape != logged_rewards.shape:
            raise ValueError("logged_arms and logged_rewards must have the same length.")
        self._k = n_arms
        self._arms = logged_arms.astype(int)
        self._rewards = logged_rewards.astype(float)

    @property
    def n_arms(self) -> int:
        return self._k

    @property
    def T(self) -> int:
        return len(self._arms)

    def reset(self) -> None:
        pass

    def query(self, t: int, proposed_arm: int) -> float | None:
        """Return reward if arm matches logged choice, else None."""
        if proposed_arm == self._arms[t]:
            return self._rewards[t]
        return None

    @property
    def logged_arms(self) -> np.ndarray:
        return self._arms

    @property
    def logged_rewards(self) -> np.ndarray:
        return self._rewards
