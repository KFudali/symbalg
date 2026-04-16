from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from algebra.core.expression import Expression
from algebra.exceptions import ShapeMismatchError

from tools.symbolic import Symbolic, SymbolicNode, BinaryOpType, UnaryOpType, op


@dataclass(frozen=True)
class ExpressionValue(op.Value[Expression]):
    def fold(self) -> np.ndarray:
        return self.value.eval()

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.value.output_shape


@dataclass(frozen=True)
class ExpressionUnaryOp(op.UnaryOp[Expression]):
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class ExpressionBinaryOp(op.BinaryOp[Expression]):
    output_shape: tuple[int, ...]


ExprNode = SymbolicNode[Expression]

class SymbolicExpression(Symbolic[Expression], Expression):
    COMPATIBLE_TYPES = (
        Expression,
        ExpressionBinaryOp,
        ExpressionUnaryOp,
        ExpressionValue,
    )

    def __init__(self, expression: ExprNode):
        Symbolic.__init__(self, expression)
        Expression.__init__(self, expression.output_shape)

    def eval(self) -> np.ndarray:
        return self.fold()

    def fold(self) -> np.ndarray:
        return super().fold()

    def copy(self) -> SymbolicExpression:
        return SymbolicExpression(self.base_op)

    def _new(self, expr: op.Op[Expression]):
        return SymbolicExpression(expr)

    def _make_value(self, operand: Expression):
        return ExpressionValue(operand)

    def _make_unary(self, optype: UnaryOpType) -> ExpressionUnaryOp:
        return ExpressionUnaryOp(optype, self.base_op, self.output_shape)

    def _make_binary(self, other: ExprNode, optype: BinaryOpType):
        return ExpressionBinaryOp(
            optype, self.base_op, self._wrap(other), self.output_shape
        )

    def _assert_compatible(self, other: Any):
        compatible_types = (SymbolicExpression, *SymbolicExpression.COMPATIBLE_TYPES)
        if isinstance(other, compatible_types):
            if self.output_shape == () or other.output_shape == ():
                return
            if self.output_shape != other.output_shape:
                raise ShapeMismatchError(
                    (
                        "SymbolicExpression can only be comined with node of equal shape",
                        f"self.output_shape: {self.output_shape}, ",
                        f"other type: {type(other)}, other.output_shape: {other.output_shape}.",
                    )
                )
        else:
            raise ValueError(
                (
                    "SymbolicExpression can only be combined with objects of type: ",
                    f"{compatible_types}.",
                )
            )
