from abc import abstractmethod, ABC
from algebra.core.space import Space, TDomain
from tools.time import TimeDim

from .dx import DxOperators
from .dt import DtOperators
from .bc_tool import BCTool

class DiscreteSpace(Space[TDomain], ABC):
    def __init__(self, domain: TDomain, time: TimeDim):
        Space.__init__(self, domain)
        self._time = time

    @property
    def time(self) -> TimeDim:
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
    def bcs(self) -> BCTool:
        pass