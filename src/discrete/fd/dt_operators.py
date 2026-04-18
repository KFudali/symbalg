from discrete.core.dt import DtOperators
from discrete.core import DiscreteTime
from algebra.symbolic import AffineOperator
from algebra.field import Field

from . import dt

class FDDtOperators(DtOperators):
    def explicit_euler(
        self, field: Field, time: DiscreteTime, order: int
    ) -> AffineOperator:
        return dt.explicit.euler(field, time, order)

    def implicit_euler(
        self, field: Field, time: DiscreteTime, order: int
    ) -> AffineOperator:
        pass
