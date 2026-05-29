from abc import abstractmethod, ABC
from typing import Generic
from .dx import DxOperators
from .dt import DtOperators
from .bc_tool import BCTool
from .discrete_time import DiscreteTime
from .domain import TDomain


class DiscreteSpace(ABC, Generic[TDomain]):
    def __init__(self, domain: TDomain):
        self._domain = domain
        self._time = DiscreteTime()

    @property
    def domain(self) -> TDomain:
        return self._domain

    @property
    def time(self) -> DiscreteTime:
        return self._time

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def dx(self) -> DxOperators:
        pass

    @property
    @abstractmethod
    def dt(self) -> DtOperators:
        pass

    @property
    @abstractmethod
    def bcs(self) -> BCTool:
        pass
