import numpy as np

from algebra.domain.operator import AffineDomainOperator
from algebra.expression import CallableExpression
from algebra.space import FieldShape
from algebra.field import Field

from discrete.core import DtOperators, DiscreteTimeView

from .operator.dt import explicit
from .domain import FDDomain


class FDDtOperators(DtOperators):
    def __init__(self, domain: FDDomain, time: DiscreteTimeView):
        self._domain = domain
        self._time = time

    def explicit(self, field: Field, order: int = 1) -> AffineDomainOperator:
        dt = CallableExpression(
            FieldShape.scalar(self._domain.space), lambda: np.array(self._time.dt())
        )
        return explicit.bfd(field, dt, order)

    def implicit(self, field: Field, order: int = 1) -> AffineDomainOperator:
        pass
