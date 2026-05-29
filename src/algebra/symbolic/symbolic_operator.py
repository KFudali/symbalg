from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generic
import numpy as np

from algebra.space import Space, ShapeTransform
from algebra.operator import Operator, TOperator
from algebra.expression import Expression, ScalarExpression
from algebra.exceptions import ShapeMismatchError

import tools.symbolic as sym
from tools.symbolic import BinaryOpType, UnaryOpType


@dataclass(frozen=True)
class OperatorNode(sym.nodes.SymbolicNode[Operator]):
    space: Space
    shape_transform: ShapeTransform


@dataclass(frozen=True)
class OperatorValueNode(sym.nodes.ValueNode[Operator], OperatorNode):
    def __init__(self, value: Operator):
        sym.nodes.ValueNode.__init__(self, value)
        OperatorNode.__init__(self, value.space, value.shape_transform)


@dataclass(frozen=True)
class ExpressionValueNode(sym.nodes.ValueNode[Expression], OperatorNode):
    """Wraps an Expression (or scalar) so it can participate in an operator tree
    via elementwise scaling. `resolve()` returns the evaluated numpy array, which
    composes correctly with Operator arithmetic (Operator * np.ndarray)."""

    def __init__(self, space: Space, value: Expression):
        sym.nodes.ValueNode.__init__(self, value)
        OperatorNode.__init__(self, space, ShapeTransform.NONE)

    def resolve(self) -> np.ndarray:
        return self.value.eval()


@dataclass(frozen=True)
class OperatorUnaryNode(sym.nodes.UnaryNode[Operator], OperatorNode):
    def __init__(self, optype: UnaryOpType, operand: OperatorNode):
        sym.nodes.UnaryNode.__init__(self, optype, operand)
        OperatorNode.__init__(self, operand.space, operand.shape_transform)


@dataclass(frozen=True)
class OperatorBinaryNode(sym.nodes.BinaryNode[Operator], OperatorNode):
    def __init__(self, optype: BinaryOpType, left: OperatorNode, right: OperatorNode):
        sym.nodes.BinaryNode.__init__(self, optype, left, right)
        OperatorNode.__init__(self, left.space, right.shape_transform)


# ---- SymbolicOperator ----------------------------------------------------------
class SymbolicOperator(sym.Symbolic[TOperator], Operator, Generic[TOperator]):
    def __init__(self, node: sym.SymbolicNode | Operator):
        sym.Symbolic.__init__(self, node)
        self._node: OperatorNode
        Operator.__init__(self, node.space, node.shape_transform)

    def copy(self) -> "SymbolicOperator":
        return SymbolicOperator(self._node)

    # ---- factory hooks ----
    def _make_value(self, operand: Any) -> sym.nodes.ValueNode:
        if isinstance(operand, Operator):
            return OperatorValueNode(operand)
        if isinstance(operand, Expression):
            return ExpressionValueNode(self.space, operand)
        if isinstance(operand, (float)):
            return ExpressionValueNode(self.space, ScalarExpression(operand))
        raise TypeError(
            f"Cannot wrap {type(operand).__name__} as an OperatorNode value."
        )

    def _make_unary(self, optype: UnaryOpType) -> sym.nodes.UnaryNode:
        return OperatorUnaryNode(optype, self._node)

    def _make_binary(self, other: Any, optype: BinaryOpType) -> sym.nodes.BinaryNode:
        return OperatorBinaryNode(optype, self._node, self._ensure_node(other))

    # ---- compatibility checks ----
    def _assert_compatible(self, other: Any, optype: BinaryOpType):
        if isinstance(other, (float, int, np.ndarray)):
            return self._assert_compatible_constants(other, optype)
        if isinstance(other, Expression):
            return self._assert_compatible_expression(other, optype)
        node = other.node if isinstance(other, SymbolicOperator) else other
        if isinstance(node, ()):
            if self.shape_transform != node.shape_transform:
                raise ShapeMismatchError(
                    "Can only combine OperatorNodes of equal shape_transform. "
                    f"self.input_shape: {self.shape_transform}, "
                    f"other.input_shape: {node.shape_transform}."
                )
            return
        comps = (
            float,
            np.ndarray,
            Expression,
            Operator,
        )
        raise ValueError(
            "SymbolicOperator can only be combined with objects of type: "
            f"{comps}. Got type {type(other)}."
        )

    def _assert_compatible_constants(
        self, other: float | np.ndarray, optype: BinaryOpType
    ):
        scaling_ops = (BinaryOpType.MUL, BinaryOpType.DIV)
        if isinstance(other, (float)):
            if optype in scaling_ops:
                return
            raise ValueError(
                f"Operator can only be scaled (MUL, DIV) by constant. Got {optype}."
            )
        if isinstance(other, np.ndarray):
            if other.shape in (self.space.shape, ()):
                raise ShapeMismatchError(
                    "Can only combine operator with Expression that space shape."
                    f"space.shape: {self.space.shape}, "
                    f"other.shape: {other.shape}."
                )
            if optype in scaling_ops:
                return
            raise ValueError(
                f"Operator can only be scaled (MUL, DIV) by array. Got {optype}."
            )

    def _assert_compatible_expression(self, other: Expression, optype: BinaryOpType):
        scaling_ops = (BinaryOpType.MUL, BinaryOpType.DIV)
        if isinstance(other, Expression):
            if other.shape in (self.space.shape, ()):
                raise ShapeMismatchError(
                    "Can only combine operator with Expression that space shape."
                    f"space.shape: {self.space.shape}, "
                    f"other.shape: {other.shape}."
                )
            if optype in scaling_ops:
                return
            raise ValueError(
                f"Operator can only be scaled (MUL, DIV) by Expression. Got {optype}."
            )
