from abc import abstractmethod, ABC
from algebra.space import TimeDim, Space, TDomain
from .dx import DxOperators
from .dt import DtOperators
from .bc_tool import BCTool

class DiscreteSpace(Space[TDomain]):
    def __init__(self, domain: TDomain, time: TimeDim):
        super().__init__(domain , time)

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
