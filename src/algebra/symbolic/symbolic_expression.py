from __future__ import annotations
from typing import Any, Self
import numpy as np

from algebra.space import FieldShape
from algebra.expression import Expression, CallableExpression
from algebra.exceptions import ShapeMismatchError

from tools.symbolic import Symbolic, BinaryOpType, nodes
from .nodes import ExpressionNode


class SymbolicExpression(Symbolic[Expression], Expression):
    def __init__(self, node: nodes.SymbolicNode[Expression], shape: FieldShape):
        Symbolic.__init__(self, node)
        Expression.__init__(self, shape)

    @classmethod
    def wrap(cls, value: Expression) -> Self:
        node = cls._make_value(value)
        return cls(node, value.fieldshape)

    @classmethod
    def _make_value(cls, other: Expression) -> nodes.ValueNode[Expression]:
        return ExpressionNode(other)

    def _ensure_node(self, other: Any) -> nodes.SymbolicNode[Expression]:
        if isinstance(other, Symbolic):
            return other.node
        if isinstance(other, nodes.SymbolicNode):
            return other
        if isinstance(other, float):
            return ExpressionNode(
                CallableExpression(
                    FieldShape.scalar(self.space), lambda: np.array(other)
                )
            )
        if isinstance(other, np.ndarray):
            return ExpressionNode(
                CallableExpression(
                    FieldShape.from_shape(self.space, other.shape),
                    lambda: other,
                )
            )
        return ExpressionNode(other)

    def eval(self) -> np.ndarray:
        return self.resolve()

    def copy(self) -> Self:
        return self.__class__(self.node, self.fieldshape)

    def _new(self, node: nodes.SymbolicNode[Expression]) -> Self:
        return self.__class__(node, self.fieldshape)

    def _combine_binary(self, other: Any, optype: BinaryOpType) -> Self:
        if not self._compatible(other, optype):
            return NotImplemented
        other_node = self._ensure_node(other)
        new_shape = self._combined_shape(other, optype)
        return self.__class__(
            nodes.BinaryNode(optype, self.node, other_node), new_shape
        )

    def _combined_shape(self, other: Any, optype: BinaryOpType) -> FieldShape:
        if isinstance(other, (Expression, np.ndarray)):
            if self.shape == ():
                return (
                    other.fieldshape
                    if isinstance(other, Expression)
                    else FieldShape.from_shape(self.space, other.shape)
                )
            return self.fieldshape
        return self.fieldshape

    def _compatible(self, other: Any, optype: BinaryOpType) -> bool:
        if isinstance(other, float):
            return True
        if isinstance(other, (Expression, np.ndarray)):
            other_shape = other.shape if isinstance(other, Expression) else other.shape
            if self.shape in ((), other_shape) or other_shape == ():
                return True
            raise ShapeMismatchError(
                f"Incompatible shape is: {self.shape} and {other_shape}"
            )
        return False
