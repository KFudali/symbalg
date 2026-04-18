from abc import ABC, abstractmethod
from algebra.field import Field
from algebra.symbolic import AffineOperator
from .discrete_time import DiscreteTime

class DtOperators(ABC):
    @abstractmethod
    def explicit_euler(
        self, field: Field, time: DiscreteTime, order: int
    ) -> AffineOperator:
        pass

    @abstractmethod
    def implicit_euler(
        self, field: Field, time: DiscreteTime, order: int
    ) -> AffineOperator:
        pass
