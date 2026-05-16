from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generic
import numpy as np

from algebra.core.operator import Operator, TOperator
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


@dataclass(frozen=True)
class OperatorNode(SymbolicNode[Operator]):
    input_shape: FieldShape
    output_shape: FieldShape


@dataclass(frozen=True)
class OperatorValueNode(ValueNode[Operator], OperatorNode):
    def __init__(self, value: Operator):
        ValueNode.__init__(self, value)
        OperatorNode.__init__(self, value.input_shape, value.output_shape)


@dataclass(frozen=True)
class ExpressionValueNode(ValueNode[Expression], OperatorNode):
    """Wraps an Expression (or scalar) so it can participate in an operator tree
    via elementwise scaling. `resolve()` returns the evaluated numpy array, which
    composes correctly with Operator arithmetic (Operator * np.ndarray)."""

    def __init__(self, value: Expression):
        ValueNode.__init__(self, value)
        OperatorNode.__init__(self, value.shape, value.shape)

    def resolve(self) -> np.ndarray:
        return self.value.eval()


@dataclass(frozen=True)
class OperatorUnaryNode(UnaryNode[Operator], OperatorNode):
    def __init__(self, optype: UnaryOpType, operand: OperatorNode):
        UnaryNode.__init__(self, optype, operand)
        OperatorNode.__init__(self, operand.input_shape, operand.output_shape)


@dataclass(frozen=True)
class OperatorBinaryNode(BinaryNode[Operator], OperatorNode):
    def __init__(self, optype: BinaryOpType, left: OperatorNode, right: OperatorNode):
        BinaryNode.__init__(self, optype, left, right)
        OperatorNode.__init__(self, left.input_shape, right.output_shape)


# ---- SymbolicOperator ----------------------------------------------------------

OPERATOR_NODES = (
    Operator,
    OperatorValueNode,
    OperatorUnaryNode,
    OperatorBinaryNode,
    ExpressionValueNode,
)


class SymbolicOperator(Symbolic[TOperator], Operator, Generic[TOperator]):

    def __init__(self, operator: SymbolicNode | Operator):
        Symbolic.__init__(self, operator)
        self._node: OperatorNode
        Operator.__init__(self, self._node.input_shape, self._node.output_shape)

    def copy(self) -> "SymbolicOperator":
        return SymbolicOperator(self._node)

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        self.resolve().apply(input_field, output_field)

    # ---- factory hooks ----

    def _make_value(self, operand: Any) -> ValueNode:
        if isinstance(operand, Operator):
            return OperatorValueNode(operand)
        if isinstance(operand, Expression):
            return ExpressionValueNode(operand)
        if isinstance(operand, (float, int, np.ndarray)):
            return ExpressionValueNode(ScalarExpression(operand))
        raise TypeError(
            f"Cannot wrap {type(operand).__name__} as an OperatorNode value."
        )

    def _make_unary(self, optype: UnaryOpType) -> OperatorUnaryNode:
        return OperatorUnaryNode(optype, self._node)

    def _make_binary(self, other: Any, optype: BinaryOpType) -> OperatorBinaryNode:
        return OperatorBinaryNode(
            optype,
            self._node,
            self._ensure_node(other),
        )

    # ---- compatibility checks ----
    def _assert_compatible(self, other: Any, optype: BinaryOpType):
        if isinstance(other, (float, int, np.ndarray)):
            return self._assert_compatible_constants(other, optype)
        if isinstance(other, Expression):
            return self._assert_compatible_expression(other, optype)
        node = other.node if isinstance(other, SymbolicOperator) else other
        if isinstance(node, OPERATOR_NODES):
            if self.input_shape != node.input_shape:
                raise ShapeMismatchError(
                    "Can only combine OperatorNodes of equal input_shape. "
                    f"self.input_shape: {self.input_shape}, "
                    f"other.input_shape: {node.input_shape}."
                )
            if self.output_shape != node.output_shape:
                raise ShapeMismatchError(
                    "Can only combine OperatorNodes of equal output_shape. "
                    f"self.output_shape: {self.output_shape}, "
                    f"other.output_shape: {node.output_shape}."
                )
            return
        comps = (
            float,
            int,
            np.ndarray,
            Expression,
            Operator,
        )
        raise ValueError(
            "SymbolicOperator can only be combined with objects of type: "
            f"{comps}. Got type {type(other)}."
        )

    def _assert_compatible_constants(
        self, other: float | int | np.ndarray, optype: BinaryOpType
    ):
        scaling_ops = (BinaryOpType.MUL, BinaryOpType.DIV)
        if isinstance(other, (float, int)):
            if optype in scaling_ops:
                return
            raise ValueError(
                f"Operator can only be scaled (MUL, DIV) by constant. Got {optype}."
            )
        if isinstance(other, np.ndarray):
            if other.shape == ():
                if optype in scaling_ops:
                    return
                raise ValueError(
                    f"Operator can only be scaled (MUL, DIV) by constant. Got {optype}."
                )
            if other.shape != self.output_shape:
                raise ShapeMismatchError(
                    "Can only combine operator with array that matches its output_shape. "
                    f"self.output_shape: {self.output_shape}, array.shape: {other.shape}."
                )
            if optype in scaling_ops:
                return
            raise ValueError(
                f"Operator can only be scaled (MUL, DIV) by array. Got {optype}."
            )

    def _assert_compatible_expression(self, other: Expression, optype: BinaryOpType):
        scaling_ops = (BinaryOpType.MUL, BinaryOpType.DIV)
        if isinstance(other, Expression):
            other_shape = other.shape
            if other_shape == ():
                if optype in scaling_ops:
                    return
                raise ValueError(
                    f"Operator can only be scaled (MUL, DIV) by Expression. Got {optype}."
                )
            if other_shape != self.output_shape:
                raise ShapeMismatchError(
                    "Can only combine operator with Expression that matches its "
                    f"output_shape. self.output_shape: {self.output_shape}, "
                    f"other.shape: {other_shape}."
                )
            if optype in scaling_ops:
                return
            raise ValueError(
                f"Operator can only be scaled (MUL, DIV) by Expression. Got {optype}."
            )
