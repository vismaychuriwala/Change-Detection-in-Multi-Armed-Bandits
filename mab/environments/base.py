from abc import ABC, abstractmethod

import numpy as np


class Environment(ABC):
    """Base class for MAB environments."""

    @property
    @abstractmethod
    def n_arms(self) -> int:
        pass

    @property
    @abstractmethod
    def T(self) -> int:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        pass
