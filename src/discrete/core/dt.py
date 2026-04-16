from abc import ABC, abstractmethod
from algebra.field import Field
from algebra.symbolic import AffineOperator
from tools.time import TimeDim

class DtOperators(ABC):
    @abstractmethod
    def explicit_euler(
        self, field: Field, time_dim: TimeDim, order: int
    ) -> AffineOperator:
        pass

    @abstractmethod
    def implicit_euler(
        self, field: Field, time_dim: TimeDim, order: int
    ) -> AffineOperator:
        pass
