from abc import ABC, abstractmethod
from algebra.operator import Operator


class DxOperators(ABC):
    @abstractmethod
    def laplace(self, order: int = 2) -> Operator:
        pass

    @abstractmethod
    def grad(self, order: int = 2) -> Operator:
        pass

    @abstractmethod
    def div(self, order: int = 2) -> Operator:
        pass
