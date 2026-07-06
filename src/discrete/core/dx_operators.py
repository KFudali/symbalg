from abc import ABC, abstractmethod
import numpy as np
from algebra.operator import Operator
from algebra.symbolic import SymbolicOperator


class DxOperators(ABC):
    def eye(self) -> SymbolicOperator:
        return SymbolicOperator.wrap(self._eye())

    def laplace(self, order: int = 2) -> SymbolicOperator:
        return SymbolicOperator.wrap(self._laplace(order))

    def grad(self, order: int = 2) -> SymbolicOperator:
        return SymbolicOperator.wrap(self._grad(order))

    def div(self, order: int = 2) -> SymbolicOperator:
        return SymbolicOperator.wrap(self._div(order))

    def array(self, weights: np.ndarray) -> SymbolicOperator:
        return SymbolicOperator.wrap(self._array(weights))

    @abstractmethod
    def _eye(self) -> Operator:
        pass

    @abstractmethod
    def _laplace(self, order: int) -> Operator:
        pass

    @abstractmethod
    def _grad(self, order: int) -> Operator:
        pass

    @abstractmethod
    def _div(self, order: int) -> Operator:
        pass

    @abstractmethod
    def _array(self, weights: np.ndarray) -> Operator:
        pass
