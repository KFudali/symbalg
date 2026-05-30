from abc import ABC, abstractmethod
from algebra.operator import Operator


class DxOperators(ABC):
    @abstractmethod
    def laplace(self) -> Operator:
        pass

    @abstractmethod
    def grad(self) -> Operator:
        pass

    @abstractmethod
    def div(self) -> Operator:
        pass
