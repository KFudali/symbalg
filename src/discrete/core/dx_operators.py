from abc import ABC, abstractmethod
import numpy as np
from algebra.domain import SymbolicDomainOperator
from algebra.space import ShapeTransform


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
    def array(
        self, weights: np.ndarray, shape_transform: ShapeTransform
    ) -> SymbolicDomainOperator: ...
