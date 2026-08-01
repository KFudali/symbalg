from __future__ import annotations

from typing import Any, Self
import numpy as np
from sparse import SparseArray
from algebra.expression.expression import (
    Expression,
    Array,
    ConstFieldExpression,
    ConstSparseExpression,
)
from algebra.exceptions import ShapeMismatchError

from algebra.space import shapes, utils


from tools.symbolic import Symbolic, BinaryOpType, nodes
from tools.symbolic.optype import MatBinOpType, MatUnOpType
from .nodes import ExpressionNode, TensorOpNode, TensorUnaryOpNode


class SymbolicExpression(Symbolic[Expression], Expression):
    def __init__(self, node: nodes.SymbolicNode[Expression], shape: shapes.Shape):
        Symbolic.__init__(self, node)
        Expression.__init__(self, shape)

    @classmethod
    def wrap(cls, value: Expression) -> Self:
        node = cls._make_value(value)
        return cls(node, value.shape)

    @classmethod
    def _make_value(cls, other: Expression) -> ExpressionNode:
        return ExpressionNode(other)

    def _ensure_node(self, other: Any) -> ExpressionNode:
        if isinstance(other, Symbolic):
            return other.node
        if isinstance(other, nodes.SymbolicNode):
            return other
        if isinstance(other, (np.ndarray, float)):
            return ExpressionNode(ConstFieldExpression(self.space, other))
        if isinstance(other, SparseArray):
            return ExpressionNode(ConstSparseExpression(self.space, other))
        return self._make_value(other)

    def eval(self) -> Array:
        return self.resolve()

    def copy(self) -> Self:
        return self.__class__(self.node, self.shape)

    def _new(self, node: nodes.SymbolicNode[Expression]) -> Self:
        return self.__class__(node, self.shape)

    def _combine_mat(self, other: Any, optype: MatBinOpType) -> Self:
        if not self._compatible_mat(other, optype):
            return NotImplemented
        other_node = self._ensure_node(other)
        new_shape = utils.project_shape(self.shape, other_node.shape, optype)
        subscripts = utils.project_einsum(self.shape, other_node.shape, optype)
        node = TensorOpNode(self.node, other_node, subscripts)
        return self.__class__(node, new_shape)

    def _combine_binary(
        self, other: Any, optype: BinaryOpType, reverse: bool = False
    ) -> Self:
        if not self._compatible(other, optype, reverse):
            return NotImplemented
        other_node = self._ensure_node(other)
        node = nodes.BinaryNode(optype, self.node, other_node)
        if self.shape.is_scalar() and isinstance(other, Expression):
            return self.__class__(node, other.shape)
        return self._new(node)

    def _compatible(
        self, other: Any, optype: BinaryOpType, reverse: bool = False
    ) -> bool:
        if isinstance(other, float):
            return True
        if isinstance(other, Expression):
            if self.shape.is_scalar() or other.shape.is_scalar():
                return True
            if self.shape == other.shape:
                return True
            raise ShapeMismatchError(
                f"Incompatible shape is: {self.shape} and {other.shape}"
            )
        if isinstance(other, (np.ndarray, SparseArray)):
            if self.shape.is_scalar() or other.shape == ():
                return True
            if self.shape.fieldshape() != other.shape:
                raise ShapeMismatchError(
                    f"Incompatible shape is: {self.shape} and {other.shape}"
                )
            return True
        return False

    def _compatible_mat(self, other: Any, optype: MatBinOpType) -> bool:
        if isinstance(other, float):
            return True
        if isinstance(other, (Expression, Array)):
            try:
                other_node = self._ensure_node(other)
                utils.project_shape(self.shape, other_node.shape, optype)
                return True
            except ShapeMismatchError:
                return False
        return False

    def _unary_mat(self, optype: MatUnOpType) -> Self:
        if optype == MatUnOpType.TRACE:
            comps = self.components
            if len(comps) != 2:
                raise ShapeMismatchError(
                    f"Cannot trace a tensor with components {comps}"
                )
            if comps[0] != comps[1]:
                raise ShapeMismatchError(
                    f"Cannot trace a non-square tensor with components {comps}"
                )
            subscripts = "aa...->..."
            new_shape = shapes.Shape(self.space, ())
            node = TensorUnaryOpNode(self.node, subscripts)
            return self.__class__(node, new_shape)
        return NotImplemented

    def dot(self, other: Expression) -> SymbolicExpression:
        return self._combine_mat(other, MatBinOpType.DOT)

    def inner(self, other: Expression) -> SymbolicExpression:
        return self._combine_mat(other, MatBinOpType.INNER)

    def outer(self, other: Expression) -> SymbolicExpression:
        return self._combine_mat(other, MatBinOpType.OUTER)

    def trace(self) -> SymbolicExpression:
        return self._unary_mat(MatUnOpType.TRACE)
