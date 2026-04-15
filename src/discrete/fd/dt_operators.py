from discrete.core.dt import DtOperators
from algebra.symbolic import AffineOperator
from algebra.field import Field

from . import dt


class FDDtOperators(DtOperators):
    def explicit_euler(self, field: Field, order: int = 1) -> AffineOperator:
        return dt.explicit.euler(field, order)

    def implicit_euler(self, field: Field, order: int = 1) -> AffineOperator:
        pass
