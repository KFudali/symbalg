from abc import abstractmethod, ABC
from typing import Generic
import numpy as np

from algebra.space import Space

from .domain import TDomain

from .dx_operators import DxOperators
from .dt_operators import DtOperators
from .discrete_time import DiscreteTime
from algebra.systems.bcs import BoundaryTool


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
    def time(self) -> DiscreteTime:
        return self._time

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
    def bc_tool(self) -> BoundaryTool:
        pass

    @abstractmethod
    def points(self) -> tuple[np.ndarray, ...]:
        pass
