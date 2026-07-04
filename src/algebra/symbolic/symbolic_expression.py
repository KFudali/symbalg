from __future__ import annotations
from typing import Any, Self
import numpy as np

from algebra.space import FieldShape
from algebra.expression import Expression, CallableExpression
from algebra.exceptions import ShapeMismatchError

from tools.symbolic import Symbolic, BinaryOpType, nodes
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

        if optype in (BinaryOpType.DOT, BinaryOpType.INNER, BinaryOpType.OUTER):
            subscripts = self._tensor_subscripts(other, optype)
            node = TensorOpNode(self.node, other_node, subscripts)
        else:
            node = nodes.BinaryNode(optype, self.node, other_node)

        return self.__class__(node, new_shape)

    def _tensor_subscripts(self, other: Any, optype: BinaryOpType) -> str:
        a_comps = self.comps
        if isinstance(other, Expression):
            b_comps = other.comps
        elif isinstance(other, np.ndarray):
            b_comps = FieldShape.from_shape(self.space, other.shape).components
        else:
            b_comps = ()

        letters = "abcdefghijklmnopqrstuvwxyz"
        n, m = len(a_comps), len(b_comps)

        if optype == BinaryOpType.DOT:
            if not a_comps or not b_comps:
                return "...,...->..."
            a_labels = letters[:n]
            b_labels = letters[n - 1 : n - 1 + m]
            result_labels = a_labels[:-1] + b_labels[1:]
            return f"{a_labels}...,{b_labels}...->{result_labels}..."

        elif optype == BinaryOpType.INNER:
            if not a_comps or not b_comps:
                return "...,...->..."
            a_labels = letters[:n]
            b_labels = letters[:n]
            return f"{a_labels}...,{b_labels}...->..."

        elif optype == BinaryOpType.OUTER:
            if not a_comps or not b_comps:
                return "...,...->..."
            a_labels = letters[:n]
            b_labels = letters[n : n + m]
            result_labels = a_labels + b_labels
            return f"{a_labels}...,{b_labels}...->{result_labels}..."

        return "...,...->..."

    def _combined_shape(self, other: Any, optype: BinaryOpType) -> FieldShape:
        if optype in (BinaryOpType.DOT, BinaryOpType.INNER, BinaryOpType.OUTER):
            return self._tensor_shape(other, optype)

        if isinstance(other, (Expression, np.ndarray)):
            if self.shape == ():
                return (
                    other.fieldshape
                    if isinstance(other, Expression)
                    else FieldShape.from_shape(self.space, other.shape)
                )
            return self.fieldshape
        return self.fieldshape

    def _tensor_shape(self, other: Any, optype: BinaryOpType) -> FieldShape:
        a_comps = self.comps
        if isinstance(other, Expression):
            b_comps = other.comps
        elif isinstance(other, np.ndarray):
            b_comps = FieldShape.from_shape(self.space, other.shape).components
        else:
            b_comps = ()

        if optype == BinaryOpType.DOT:
            if not a_comps or not b_comps:
                return FieldShape(self.space, a_comps or b_comps)
            return FieldShape(self.space, a_comps[:-1] + b_comps[1:])

        elif optype == BinaryOpType.INNER:
            if not a_comps or not b_comps:
                return FieldShape(self.space, a_comps or b_comps)
            return FieldShape(self.space, ())

        elif optype == BinaryOpType.OUTER:
            return FieldShape(self.space, a_comps + b_comps)

        return self.fieldshape

    def _compatible(self, other: Any, optype: BinaryOpType) -> bool:
        if optype == BinaryOpType.DOT:
            if isinstance(other, float):
                return True
            if isinstance(other, (Expression, np.ndarray)):
                b_comps = (
                    other.comps
                    if isinstance(other, Expression)
                    else other.shape[: -self.space.ndim]
                )
                a_comps = self.comps
                if not a_comps or not b_comps:
                    return True
                if a_comps[-1] == b_comps[0]:
                    return True
                raise ShapeMismatchError(
                    f"Cannot dot: incompatible shapes {a_comps} and {b_comps}"
                )
            return False

        if optype == BinaryOpType.INNER:
            if isinstance(other, float):
                return True
            if isinstance(other, (Expression, np.ndarray)):
                b_comps = (
                    other.comps
                    if isinstance(other, Expression)
                    else other.shape[: -self.space.ndim]
                )
                a_comps = self.comps
                if not a_comps or not b_comps:
                    return True
                if a_comps == b_comps:
                    return True
                raise ShapeMismatchError(
                    f"Cannot inner: incompatible shapes {a_comps} and {b_comps}"
                )
            return False

        if optype == BinaryOpType.OUTER:
            return isinstance(other, (Expression, np.ndarray, float))

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

    def dot(self, other: Expression) -> SymbolicExpression:
        return self._combine_binary(other, BinaryOpType.DOT)

    def inner(self, other: Expression) -> SymbolicExpression:
        return self._combine_binary(other, BinaryOpType.INNER)

    def outer(self, other: Expression) -> SymbolicExpression:
        return self._combine_binary(other, BinaryOpType.OUTER)
