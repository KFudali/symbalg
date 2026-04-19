from abc import ABC, abstractmethod
from algebra.field import Field
from algebra.symbolic import AffineOperator
from .discrete_time import DiscreteTime

class DtOperators(ABC):
    @abstractmethod
    def euler(self, field: Field, time: DiscreteTime) -> AffineOperator:
        pass

    @abstractmethod
    def bfd(self, field: Field, time: DiscreteTime, order: int = 1) -> AffineOperator:
        pass
