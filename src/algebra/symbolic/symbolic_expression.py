from __future__ import annotations

from typing import Any, Self
import numpy as np

from algebra.space import FieldShape, utils
from algebra.expression import Expression, CallableExpression, ConstExpression
from algebra.exceptions import ShapeMismatchError

from tools.symbolic import Symbolic, BinaryOpType, nodes
from tools.symbolic.optype import MatOpType
from .nodes import ExpressionNode, TensorOpNode


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
        if isinstance(other, (np.ndarray, float)):
            return ExpressionNode(ConstExpression(self.space, other))
        return self._make_value(other)

    def eval(self) -> np.ndarray:
        return self.resolve()

    def copy(self) -> Self:
        return self.__class__(self.node, self.fieldshape)

    def _new(self, node: nodes.SymbolicNode[Expression]) -> Self:
        return self.__class__(node, self.fieldshape)

    def _combine_mat(self, other: Any, optype: MatOpType) -> Self:
        if not self._compatible_mat(other, optype):
            return NotImplemented

    def _combine_binary(
        self, other: Any, optype: BinaryOpType, reverse: bool = False
    ) -> Self:
        if not self._compatible(other, optype, reverse):
            return NotImplemented
        other_node = self._ensure_node(other)
        return self._new(nodes.BinaryNode(optype, self.node, other_node))

    def _compatible(
        self, other: Any, optype: BinaryOpType, reverse: bool = False
    ) -> bool:
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

    def _compatible_mat(self, other: Any, optype: MatOpType) -> bool:
        pass

    def dot(self, other: Expression) -> SymbolicExpression:
        return self._combine_mat(other, MatOpType.DOT)

    def inner(self, other: Expression) -> SymbolicExpression:
        return self._combine_mat(other, MatOpType.INNER)

    def outer(self, other: Expression) -> SymbolicExpression:
        return self._combine_mat(other, MatOpType.OUTER)
