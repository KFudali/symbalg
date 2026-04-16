from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generic
import numpy as np

from algebra.core.operator import Operator, TOperator
from algebra.exceptions import ShapeMismatchError
from algebra.core.expression import ScalarExpression

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
class ScalarExpOperatorValue(op.Value[ScalarExpression]):
    @property
    def input_shape(self) -> tuple[int, ...]:
        return ()
    @property
    def output_shape(self) -> tuple[int, ...]:
        return ()

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
        ScalarExpression,
        float, np.ndarray
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

    def _make_value(self, operand: Operator | ScalarExpression):
        if isinstance(operand, ScalarExpression):
            return ScalarExpOperatorValue(operand)
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

    def _assert_compatible(self, other: Any):
        compatible_types = (SymbolicOperator, * SymbolicOperator.COMPATIBLE_TYPES)
        if isinstance(other, (float, ScalarExpression)):
            return
        if isinstance(other, np.ndarray):
            if other.shape != self.output_shape:
                raise ShapeMismatchError((
                    "Can only combine operator with array that matches its output_shape. ",
                    f"self.output_shape: {self.output_shape}, array.shape: {other.shape}."
                ))
        if isinstance(other, compatible_types):
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
            raise ValueError((
                "SymbolicOperator can only be combined with objects of type: ",
                f"{compatible_types}. \n. Got type {type(other)}"
            ))
