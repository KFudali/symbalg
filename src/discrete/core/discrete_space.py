from abc import ABC, abstractmethod
from typing import Generic
from .domain import TDomain
from .dx import DxOperators
from .dt import DtOperators
from .bc_tool import BCTool


class DiscreteSpace(ABC, Generic[TDomain]):
    def __init__(self, domain: TDomain):
        self._domain = domain

    def domain(self) -> TDomain:
        return self._domain

    @abstractmethod
    def dx(self) -> DxOperators:
        pass

    @abstractmethod
    def dt(self) -> DtOperators:
        pass

    @abstractmethod
    def bc_tool(self) -> BCTool:
        pass
