from __future__ import annotations
from typing import Self
import numpy as np

from algebra.expression import Expression
from algebra.operator import AffineOperator, ArrayOperator
from algebra.domain.bcs import BoundaryCondition
from .domain_operator import DomainOperator
from .symbolic_domain_operator import SymbolicDomainOperator


class AffineDomainOperator(AffineOperator[SymbolicDomainOperator], DomainOperator):
    def __init__(
        self,
        operator: DomainOperator,
        expression: Expression,
    ):
        AffineOperator.__init__(self, operator, expression)
        DomainOperator.__init__(self, operator.domain, operator.shape_transform)

    @classmethod
    def _wrap_operator(cls, operator: DomainOperator) -> SymbolicDomainOperator:
        if not isinstance(operator, SymbolicDomainOperator):
            return SymbolicDomainOperator.wrap(operator)
        return operator

    def as_array(self) -> ArrayOperator:
        raise ValueError(
            (
                "Affine operator cannot be represented using single array. ",
                "Create new AffineOperator using this.operator.as_array(), this.expresion.",
            )
        )

    def apply_bcs(self, bcs: list[BoundaryCondition], rhs: np.ndarray) -> Self:
        return self.__class__(self.operator.apply_bcs(bcs, rhs), self.expression)
