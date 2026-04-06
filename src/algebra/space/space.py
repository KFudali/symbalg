from abc import ABC, abstractmethod
from typing import Generic
from .domain import TDomain
import numpy as np


class Space(ABC, Generic[TDomain]):

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def domain(self) -> TDomain:
        pass

    @abstractmethod
    def flatten(self, field: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reshape(self, field: np.ndarray, components: int) -> np.ndarray:
        pass
