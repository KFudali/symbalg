from discrete.core.dt import DtOperators
from algebra.symbolic import AffineOperator
from algebra.field import Field

from . import dt



class FDDtOperators(DtOperators):
    def explicit_euler(
        self, field: Field, time_dim: TimeDim, order: int
    ) -> AffineOperator:
        return dt.explicit.euler(field, time_dim, order)

    def implicit_euler(
        self, field: Field, time_dim: TimeDim, order: int
    ) -> AffineOperator:
