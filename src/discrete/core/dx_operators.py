from abc import ABC, abstractmethod
import numpy as np
from algebra.domain import DomainOperator


class DxOperators(ABC):
    @abstractmethod
    def eye(self) -> DomainOperator: ...

    @abstractmethod
    def laplace(self, order: int = 2) -> DomainOperator: ...

    @abstractmethod
    def grad(self, order: int = 2) -> DomainOperator: ...

    @abstractmethod
    def div(self, order: int = 2) -> DomainOperator: ...

    @abstractmethod
    def array(self, weights: np.ndarray) -> DomainOperator: ...
