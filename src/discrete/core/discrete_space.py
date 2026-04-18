from abc import abstractmethod, ABC
from algebra.core.space import Space, TDomain

from .dx import DxOperators
from .dt import DtOperators
from .bc_tool import BCTool
from .discrete_time import DiscreteTime

class DiscreteSpace(Space[TDomain], ABC):
    def __init__(self, domain: TDomain):
        time = DiscreteTime()
        Space.__init__(self, domain, time)

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
    def bcs(self) -> BCTool:
        pass
