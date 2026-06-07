from discrete.core import DtOperators, DiscreteTimeView
from algebra.symbolic import AffineOperator
from algebra.expression import CallableScalarExpression
from algebra.space import Space
from algebra.field import Field
from .operators.dt import explicit


class FDDtOperators(DtOperators):
    def __init__(self, space: Space, time: DiscreteTimeView):
        self._space = space
        self._time = time

    def explicit(self, field: Field, order: int = 1) -> AffineOperator:
        dt = CallableScalarExpression(self._time.dt)
        return explicit.bfd(field, dt, order)

    def implicit(self, field: Field, order: int = 1) -> AffineOperator:
        pass
