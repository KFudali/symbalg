from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generic
import numpy as np

from algebra.core.operator import Operator, TOperator
from algebra.exceptions import ShapeMismatchError
from algebra.core.expression import Expression

from tools.symbolic import Symbolic, SymbolicNode, BinaryOpType, UnaryOpType, op

@dataclass(frozen=True)
class OperatorValue(op.Value[Operator]):
    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.value.input_shape
    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.value.output_shape

@dataclass(frozen=True)
class ExpressionValue(op.Value[Expression]):
    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.value.output_shape
    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.value.output_shape
    def fold(self) -> np.ndarray:
        return self.value.eval()

@dataclass(frozen=True)
class OperatorUnaryOp(op.UnaryOp[Operator]):
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

@dataclass(frozen=True)
class OperatorBinaryOp(op.BinaryOp[Operator]):
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

OperatorNode = SymbolicNode[Operator]

class SymbolicOperator(Symbolic[TOperator], Operator, Generic[TOperator]):
    COMPATIBLE_TYPES = (
        Operator,
        OperatorBinaryOp,
        OperatorUnaryOp,
        OperatorValue,
    )

    def __init__(self, operator: OperatorNode):
        Symbolic.__init__(self, operator)
        Operator.__init__(self, operator.input_shape, operator.output_shape)

    def copy(self) -> SymbolicOperator:
        return SymbolicOperator(self.base_op)

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        self.fold().apply(input_field, output_field)

    def _new(self, expr: op.Op[Operator]):
        return SymbolicOperator(expr)

    def _make_value(self, operand: Operator | Expression):
        if isinstance(operand, Expression):
            return ExpressionValue(operand)
        return OperatorValue(operand)

    def _make_unary(self, optype: UnaryOpType) -> OperatorUnaryOp:
        return OperatorUnaryOp(
            optype, self.base_op, self.input_shape, self.output_shape
        )

    def _make_binary(self, other: OperatorNode, optype: BinaryOpType):
        return OperatorBinaryOp(
            optype, self.base_op, self._wrap(other),
            self.input_shape, self.output_shape
        )

    def _assert_compatible(self, other: Any, optype: BinaryOpType):
        if isinstance(other, float):
            if optype in [BinaryOpType.MUL, BinaryOpType.DIV]:
                return
            raise ValueError((
                f"Operator can only by scaled (MUL, DIV) by constant. Got {optype}."
            ))
        if isinstance(other, np.ndarray):
            if other.shape == ():
                if optype in [BinaryOpType.MUL, BinaryOpType.DIV]:
                    return
                raise ValueError((
                    f"Operator can only by scaled (MUL, DIV) by constant. Got {optype}."
                ))
            if other.shape != self.output_shape:
                raise ShapeMismatchError((
                    "Can only combine operator with array that matches its output_shape. ",
                    f"self.output_shape: {self.output_shape}, array.shape: {other.shape}."
                ))
            if optype in [BinaryOpType.MUL, BinaryOpType.DIV]:
                return
            raise ValueError((
                f"Operator can only by scaled (MUL, DIV) by constant. Got {optype}."
            ))
        if isinstance(other, Expression):
            if other.output_shape == ():
                if optype in [BinaryOpType.MUL, BinaryOpType.DIV]:
                    return
                raise ValueError((
                    f"Operator can only by scaled (MUL, DIV) by Expression. Got {optype}."
                ))
            if other.output_shape != self.output_shape:
                raise ShapeMismatchError((
                    "Can only combine operator with Expression that matches its ",
                    f"output_shape. self.output_shape: {self.output_shape}, ",
                    f"other.output_shape: {other.output_shape}."
                ))
            if optype in [BinaryOpType.MUL, BinaryOpType.DIV]:
                return
            raise ValueError((
                f"Operator can only by scaled (MUL, DIV) by constant. Got {optype}."
            ))
        if isinstance(other, SymbolicOperator.COMPATIBLE_TYPES):
            if self.input_shape != other.input_shape:
                raise ShapeMismatchError((
                    "Can only combine OperatorNodes of equal input_shape.",
                    f"self.input_shape: {self.input_shape},",
                    f"other.input_shape: {other.input_shape}."
                ))
            if self.output_shape != other.output_shape:
                raise ShapeMismatchError((
                    "Can only combine OperatorNodes of equal output_shape.",
                    f"self.output_shape: {self.output_shape},",
                    f"other.output_shape: {other.output_shape}."
                ))
        else:
            compatible = (
                float, np.ndarray, Expression, *SymbolicOperator.COMPATIBLE_TYPES
            )
            raise ValueError((
                "SymbolicOperator can only be combined with objects of type: ",
                f"{compatible}. \n. Got type {type(other)}"
            ))
