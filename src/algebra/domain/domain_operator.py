from __future__ import annotations
from typing import Self, Generic
from abc import ABC, abstractmethod
import numpy as np


from tools.symbolic import nodes
from algebra.space import FieldShape, ShapeTransform
from algebra.field import Field


from algebra.operator import TOperator, ArrayOperator
from algebra.operator.symbolic import SymbolicOperator
from algebra.expression.symbolic import SymbolicExpression

from algebra.expression import CallableExpression


from .bcs import BoundaryCondition
from .domain import TDomain


class DomainOperator(SymbolicOperator[TOperator], ABC, Generic[TDomain, TOperator]):
    def __init__(
        self,
        node: nodes.SymbolicNode[TOperator],
        domain: TDomain,
        shape_transform: ShapeTransform,
    ):
        super().__init__(node, domain.space, shape_transform)
        self._domain = domain

    def _new(self, node: nodes.SymbolicNode[TOperator]) -> Self:
        return self.__class__(node, self.domain, self.shape_transform)

    @classmethod
    def wrap(cls, value: TOperator, domain: TDomain) -> Self:
        node = cls._make_value(value)
        return cls(node, domain, value.shape_transform)

    def copy(self) -> Self:
        return self.__class__(self.node, self.domain, self.shape_transform)

    @property
    def domain(self) -> TDomain:
        return self._domain

    def apply_bcs(self, bcs: list[BoundaryCondition], rhs: np.ndarray) -> Self:
        return self.__class__(
            self.domain.boundary_tool.apply_bcs(bcs, self, rhs),
            self.domain,
            self.shape_transform,
        )

    @abstractmethod
    def as_array(self) -> ArrayOperator:
        pass

    def of(self, field: Field) -> SymbolicExpression:
        def apply_to_field():
            return self.apply_to(field.value().eval())

        out_shape = self.shape_transform.transform(self.space, field.shape)
        result = CallableExpression(
            FieldShape.from_shape(self.space, out_shape), apply_to_field
        )
        return SymbolicExpression.wrap(result)
