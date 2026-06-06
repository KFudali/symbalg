from abc import ABC, abstractmethod
from algebra.field import Field
from algebra.symbolic import AffineOperator


class DtOperators(ABC):
    def euler(self, field: Field) -> AffineOperator:
        return self._euler(field)

    @abstractmethod
    def _euler(self, field: Field) -> AffineOperator:
        pass
