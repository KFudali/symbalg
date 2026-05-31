from abc import ABC, abstractmethod
from algebra.field import Field
from algebra.symbolic import AffineOperator


class DtOperators(ABC):
    @abstractmethod
    def euler(self, field: Field) -> AffineOperator:
        pass
