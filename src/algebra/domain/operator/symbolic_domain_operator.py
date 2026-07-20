from __future__ import annotations
from typing import Self

from tools.symbolic import nodes
from algebra.space import ShapeTransform

from algebra.operator.symbolic import SymbolicOperator

from algebra.domain.domain import TDomain
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
