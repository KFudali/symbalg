from abc import ABC, abstractmethod
from algebra.operator import Operator
from algebra.symbolic import SymbolicOperator

class DxOperators(ABC):
    def laplace(self, order: int = 2) -> SymbolicOperator:
        return SymbolicOperator.wrap(self._laplace(order))

    def grad(self, order: int = 2) -> SymbolicOperator:
        return SymbolicOperator.wrap(self._grad(order))

    def div(self, order: int = 2) -> SymbolicOperator:
        return SymbolicOperator.wrap(self._div(order))

    @abstractmethod
    def _laplace(self, order: int) -> Operator:
        pass

    @abstractmethod
    def _grad(self, order: int) -> Operator:
        pass

    @abstractmethod
    def _div(self, order: int) -> Operator:
        pass
