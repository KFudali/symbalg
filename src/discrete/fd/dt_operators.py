import numpy as np
from algebra.expression import CallableExpression
from discrete.core import DtOperators, DiscreteTimeView
from algebra.symbolic import AffineOperator
from algebra.space import Space, FieldShape
from algebra.field import Field
from .operators.dt import explicit


class FDDtOperators(DtOperators):
    def __init__(self, space: Space, time: DiscreteTimeView):
        self._space = space
        self._time = time

    def explicit(self, field: Field, order: int = 1) -> AffineOperator:
        dt = CallableExpression(
            FieldShape.scalar(self._space), lambda: np.array(self._time.dt())
        )
        return explicit.bfd(field, dt, order)

    def implicit(self, field: Field, order: int = 1) -> AffineOperator:
        pass
