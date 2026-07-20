from abc import ABC, abstractmethod
import numpy as np
from algebra.domain import SymbolicDomainOperator


class DxOperators(ABC):
    @abstractmethod
    def eye(self) -> SymbolicDomainOperator: ...

    @abstractmethod
    def laplace(self, order: int = 2) -> SymbolicDomainOperator: ...

    @abstractmethod
    def grad(self, order: int = 2) -> SymbolicDomainOperator: ...

    @abstractmethod
    def div(self, order: int = 2) -> SymbolicDomainOperator: ...

    @abstractmethod
    def array(self, weights: np.ndarray) -> SymbolicDomainOperator: ...
