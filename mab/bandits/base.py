from abc import ABC, abstractmethod


class BanditAlgorithm(ABC):
    """Base class for bandit algorithms."""

    @abstractmethod
    def select_arm(self) -> int:
        pass

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
