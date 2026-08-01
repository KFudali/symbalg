from __future__ import annotations
from typing import Self
import numpy as np
from tools.symbolic import nodes
from algebra.space import ShapeTransform
from algebra.domain import TDomain
from algebra.domain.bcs import BoundaryCondition
from algebra.operator import ArrayOperator
from algebra.operator.symbolic import SymbolicOperator
from .domain_operator import DomainOperator


class SymbolicDomainOperator(
    SymbolicOperator[DomainOperator[TDomain]], DomainOperator[TDomain]
):
    def __init__(
        self,
        node: nodes.SymbolicNode[DomainOperator[TDomain]],
        shape_transform: ShapeTransform,
        domain: TDomain,
    ):
        SymbolicOperator.__init__(self, node, domain.space, shape_transform)
        DomainOperator.__init__(self, domain, shape_transform)

    @classmethod
    def wrap(cls, value: DomainOperator) -> Self:
        node = cls._make_value(value)
        return cls(node, value.shape_transform, value.domain)

    def _new(self, node: nodes.SymbolicNode[DomainOperator]) -> Self:
        return self.__class__(node, self.shape_transform, self._domain)

    def copy(self) -> Self:
        return self.__class__(self.node, self.shape_transform, self._domain)

    def as_array(self) -> ArrayOperator:
        return self.resolve().as_array()

    def apply_bcs(self, bcs: list[BoundaryCondition], rhs: np.ndarray) -> Self:
        return self.resolve().apply_bcs(bcs, rhs)
