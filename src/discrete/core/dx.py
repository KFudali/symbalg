from abc import ABC, abstractmethod
from algebra.operator import Operator

class DxOperators(ABC):
    @abstractmethod
    def laplace(self, components: int) -> Operator:
        pass

    @abstractmethod
    def grad(self, components: int) -> Operator:
        pass

    @abstractmethod
    def div(self, components: int) -> Operator:
        pass
