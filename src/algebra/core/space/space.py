from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import numpy as np

from .domain import TDomain
from .time_dim import TimeDim

class Space(ABC, Generic[TDomain]):
    def __init__(self, domain: TDomain, time: TimeDim):
        self._domain = domain
        self._time = time

    @property
    def time(self) -> TimeDim:
        return self._time

    @property
    def domain(self) -> TDomain:
        return self._domain

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def points(self) -> np.ndarray:
        pass

TSpace = TypeVar("TSpace", bound=Space)


