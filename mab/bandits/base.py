from abc import ABC, abstractmethod


class BanditAlgorithm(ABC):
    """Abstract base class for bandit algorithms.

    The experiment runner calls select_arm() to get the arm to play,
    then calls update(arm, reward) after observing the reward.
    Regret is accumulated externally by the experiment runner using
    the true arm means returned by the environment.
    """

    @abstractmethod
    def select_arm(self) -> int:
        """Return the index of the arm to play."""

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Update internal state after playing arm and observing reward."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state to start a fresh trial."""
