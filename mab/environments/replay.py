"""Logged (offline) environment for rejection-sampling bandit evaluation.

When reward data comes from real-world logs only the arm that was actually
displayed has an observed reward.  LoggedEnv wraps such a dataset.

At each timestep the caller proposes an arm via query(); if it matches the
logged arm the reward is returned (a *valid* event).  Otherwise None is
returned, signalling that the event should be skipped and the bandit should
NOT be updated.

This is the rejection-sampling evaluator of Li et al. (2010) and the offline
evaluation protocol used by Liu et al. (2018) for Yahoo! R6A.

Reference
---------
Li, L. et al. (2010).  A Contextual-Bandit Approach to Personalized News
Article Recommendation.  WWW 2010.
"""

import numpy as np


class LoggedEnv:
    """Offline bandit environment backed by a logged (arm, reward) dataset.

    Parameters
    ----------
    n_arms : int
        Total number of arms.
    logged_arms : ndarray, shape (T,)
        Which arm was displayed at each timestep.
    logged_rewards : ndarray, shape (T,)
        Observed reward for the logged arm at each timestep.
    """

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

    # --- Properties that mirror the Environment ABC so bandits/scripts work uniformly ---

    @property
    def n_arms(self) -> int:
        return self._k

    @property
    def T(self) -> int:
        return len(self._arms)

    def reset(self) -> None:
        """No mutable state — reset is a no-op."""

    # --- Offline query interface ---

    def query(self, t: int, proposed_arm: int) -> float | None:
        """Return reward if proposed_arm matches the logged arm at step t.

        Returns None to signal a skipped (invalid) event.  The bandit must
        NOT be updated on a skipped event.
        """
        if proposed_arm == self._arms[t]:
            return self._rewards[t]
        return None

    @property
    def logged_arms(self) -> np.ndarray:
        return self._arms

    @property
    def logged_rewards(self) -> np.ndarray:
        return self._rewards
