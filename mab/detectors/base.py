from abc import ABC, abstractmethod


class ChangeDetector(ABC):
    """Abstract base class for online change detection algorithms.

    Subclasses maintain internal state and consume one reward at a time.
    Call reset() to clear state (e.g., after the bandit resets an arm).
    """

    @abstractmethod
    def update(self, reward: float) -> bool:
        """Process a new observation.

        Returns True if a change is detected (alarm), False otherwise.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state as if no observations have been seen."""
