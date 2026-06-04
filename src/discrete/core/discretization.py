from abc import abstractmethod, ABC
from typing import Generic
from algebra.space import Space
from algebra.operator import Operator
from .domain import TDomain, Boundary
from .dx import DxOperators
from .dt import DtOperators
from .bcs import BCTool
from .discrete_time import DiscreteTime


class Discretization(ABC, Generic[TDomain]):
    def __init__(self, space: Space, domain: TDomain):
        self._space = space
        self._domain = domain
        self._time = DiscreteTime(dt=0.01)

    @property
    def space(self) -> Space:
        return self._space

    @property
    def domain(self) -> TDomain:
        return self._domain

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
    def bc_tool(self) -> BCTool[Operator, Boundary]:
        pass
