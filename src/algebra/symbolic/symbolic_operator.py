from __future__ import annotations
from typing import Any, Self, TYPE_CHECKING
import numpy as np

from algebra.space import Space, ShapeTransform
from algebra.operator import Operator, TOperator
from algebra.expression import Expression, ScalarExpression

from tools.symbolic import Symbolic, BinaryOpType, nodes
from .nodes import ExprScaleNode

if TYPE_CHECKING:
    from algebra.field import Field
    from .symbolic_expression import SymbolicExpression


class SymbolicOperator(Symbolic[TOperator], Operator):
    def __init__(
        self,
        node: nodes.SymbolicNode[Operator],
        space: Space,
        shape_transform: ShapeTransform,
    ):
        Symbolic.__init__(self, node)
        Operator.__init__(self, space, shape_transform)

    def of(self, field: "Field") -> "SymbolicExpression":
        from .symbolic_expression import SymbolicExpression

        return SymbolicExpression.wrap(Operator.of(self, field))

    def apply(self, inp: np.ndarray, out: np.ndarray):
        self.resolve().apply(inp, out)

    def _scale(self, other: float) -> Self:
        raise ValueError("SymbolicOperator should not use _scale method")

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        raise ValueError("SymbolicOperator should not use _combine method")

    @classmethod
    def wrap(cls, value: TOperator) -> Self:
        node = cls._make_value(value)
        return cls(node, value.space, value.shape_transform)

    def copy(self) -> Self:
        return self.__class__(self.node, self.space, self.shape_transform)

    @classmethod
    def _make_value(cls, other: Any) -> nodes.ValueNode:
        if isinstance(other, ScalarExpression):
            return ExprScaleNode(other)
        return super()._make_value(other)

    def _new(self, node: nodes.SymbolicNode[TOperator]) -> Self:
        return self.__class__(node, self.space, self.shape_transform)

    def _compatible(self, other: Any, optype: BinaryOpType) -> bool:
        if isinstance(other, Operator):
            matches_space = self.space == other.space
            matches_shape_transform = self.shape_transform == other.shape_transform
            if matches_space and matches_shape_transform:
                return True
        is_scale = optype in (BinaryOpType.DIV, BinaryOpType.MUL)
        if isinstance(other, Expression):
            if other.shape == () and is_scale:
                return True
        if isinstance(other, float):
            if is_scale:
                return True
        return False
