from abc import ABC, abstractmethod


class ChangeDetector(ABC):
    """Base class for change detection algorithms."""

    @abstractmethod
    def update(self, reward: float) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
