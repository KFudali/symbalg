from abc import ABC, abstractmethod
from algebra.field import Field
from algebra.symbolic import AffineOperator

class DtOperators(ABC):
    @abstractmethod
    def explicit_euler(self, field: Field, order: int) -> AffineOperator:
        pass

    @abstractmethod
    def implicit_euler(self, field: Field, order: int) -> AffineOperator:
        pass
