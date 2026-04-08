from abc import ABC, abstractmethod
from typing import Generic
from .domain import TDomain
from .time import TimeSeries

class Space(ABC, Generic[TDomain]):
    def __init__(self, domain: TDomain, time: TimeSeries):
        self._domain = domain
        self._time = time

    @property
    def domain(self) -> TDomain:
        return self._domain

    @property
    @abstractmethod
    def time(self) -> TimeSeries:
        return self._time

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass
