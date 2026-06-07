from abc import ABC, abstractmethod
from algebra.field import Field
from algebra.symbolic import AffineOperator


class DtOperators(ABC):
    @abstractmethod
    def explicit(self, field: Field, order: int = 1) -> AffineOperator:
        pass

    @abstractmethod
    def implicit(self, field: Field, order: int = 1) -> AffineOperator:
        pass
