import numpy as np


from algebra.expression import CallableExpression
from algebra.space import FieldShape
from algebra.field import Field
from algebra.symbolic import AffineOperator

from .domain import FDDomain
from discrete.core import DtOperators, DiscreteTimeView

from .operator.dt import explicit


class FDDtOperators(DtOperators):
    def __init__(self, domain: FDDomain, time: DiscreteTimeView):
        self._domain = domain
        self._time = time

    def explicit(self, field: Field, order: int = 1) -> AffineOperator:
        dt = CallableExpression(
            FieldShape.scalar(self._domain.space), lambda: np.array(self._time.dt())
        )
        return explicit.bfd(field, dt, order)

    def implicit(self, field: Field, order: int = 1) -> AffineOperator:
        pass
