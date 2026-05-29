from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from algebra.core.expression import Expression, ScalarExpression
from algebra.exceptions import ShapeMismatchError
from algebra.fieldshape import FieldShape

from tools.symbolic import (
    Symbolic,
    SymbolicNode,
    BinaryOpType,
    UnaryOpType,
)
from tools.symbolic.symbolic_node import ValueNode, UnaryNode, BinaryNode

# ---- Expression-flavoured nodes ------------------------------------------------


@dataclass(frozen=True)
class ExpressionNode(SymbolicNode[Expression]):
    shape: FieldShape


@dataclass(frozen=True)
class ExpressionValueNode(ValueNode[Expression], ExpressionNode):
    """`resolve()` evaluates the wrapped Expression to a numpy array so the rest
    of the tree composes via numpy arithmetic."""

    def __init__(self, value: Expression):
        ValueNode.__init__(self, value)
        ExpressionNode.__init__(self, value.shape)

    def resolve(self) -> np.ndarray:
        return self.value.eval()


@dataclass(frozen=True)
class ExpressionUnaryNode(UnaryNode[Expression], ExpressionNode):
    def __init__(self, optype: UnaryOpType, operand: ExpressionNode):
        UnaryNode.__init__(self, optype, operand)
        ExpressionNode.__init__(self, operand.shape)


@dataclass(frozen=True)
class ExpressionBinaryNode(BinaryNode[Expression], ExpressionNode):
    def __init__(
        self, optype: BinaryOpType, left: ExpressionNode, right: ExpressionNode
    ):
        BinaryNode.__init__(self, optype, left, right)
        shape = left.shape if left.shape != () else right.shape
        ExpressionNode.__init__(self, shape)


# ---- SymbolicExpression --------------------------------------------------------


class SymbolicExpression(Symbolic[Expression], Expression):
    COMPATIBLE_NODE_TYPES = (
        ExpressionValueNode,
        ExpressionUnaryNode,
        ExpressionBinaryNode,
    )

    def __init__(self, expression: ExpressionNode | Expression):
        Symbolic.__init__(self, expression)
        self._node: ExpressionNode
        Expression.__init__(self, self._node.shape)

    def eval(self) -> np.ndarray:
        return np.asarray(self.resolve())

    def copy(self) -> "SymbolicExpression":
        return SymbolicExpression(self._node)

    # ---- factory hooks ----

    def _make_value(self, operand: Any) -> ExpressionValueNode:
        if isinstance(operand, (float, int, np.ndarray)):
            return ExpressionValueNode(ScalarExpression(operand))
        if isinstance(operand, Expression):
            return ExpressionValueNode(operand)
        raise TypeError(
            f"Cannot wrap {type(operand).__name__} as an ExpressionNode value."
        )

    def _make_unary(self, optype: UnaryOpType) -> ExpressionUnaryNode:
        return ExpressionUnaryNode(optype, self._node)

    def _make_binary(self, other: Any, optype: BinaryOpType) -> ExpressionBinaryNode:
        return ExpressionBinaryNode(optype, self._node, self._ensure_node(other))

    # ---- compatibility checks ----

    def _assert_compatible(self, other: Any, optype: BinaryOpType):
        if isinstance(other, (float, int, np.ndarray)):
            return

        node = other.node if isinstance(other, SymbolicExpression) else other
        if isinstance(other, Expression) or isinstance(
            node, SymbolicExpression.COMPATIBLE_NODE_TYPES
        ):
            other_shape = other.shape if isinstance(other, Expression) else node.shape
            if self.shape == () or other_shape == ():
                return
            if self.shape != other_shape:
                raise ShapeMismatchError(
                    "SymbolicExpression can only be combined with node of equal shape. "
                    f"self.shape: {self.shape}, "
                    f"other type: {type(other)}, other.shape: {other_shape}."
                )
            return

        compatible = (float, int, np.ndarray, Expression, SymbolicExpression)
        raise ValueError(
            "SymbolicExpression can only be combined with objects of type: "
            f"{compatible}. Got {type(other)}."
        )
