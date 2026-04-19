from discrete.core.dt import DtOperators
from discrete.core import DiscreteTime
from algebra.symbolic import AffineOperator
from algebra.field import Field

from . import dt

class FDDtOperators(DtOperators):
    def euler(self, field: Field, time: DiscreteTime) -> AffineOperator:
        return dt.explicit.euler(field, time)

    def bfd(
        self, field: Field, time: DiscreteTime, order: int
    ) -> AffineOperator:
        pass
