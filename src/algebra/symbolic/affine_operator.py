from __future__ import annotations
from typing import Callable, Any, Self, Generic
import numpy as np

from algebra.core.operator import Operator, TOperator
from algebra.core.expression import Expression, TExpression, ScalarExpression
from tools.symbolic import BinaryOpType, BINARY_OPS

from .symbolic_expression import SymbolicExpression
from .symbolic_operator import SymbolicOperator


class AffineOperator(Operator, Generic[TOperator, TExpression]):
    def __init__(
        self,
        operator: TOperator,
        expression: TExpression,
    ):
        self._operator = SymbolicOperator(operator)
        self._expression = SymbolicExpression(expression)
        super().__init__(operator.input_shape, operator.output_shape)

    def copy(self) -> Self:
        return self._new(self.operator, self.expression)

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        self.operator.apply(input_field, output_field)
        output_field += self.expression.eval()

    def _new(self, op: Operator, exp: Expression) -> Self:
        return type(self)(op, exp)

    @property
    def operator(self) -> SymbolicOperator[TOperator]:
        return self._operator

    @property
    def expression(self) -> SymbolicExpression:
        return self._expression

    def _combine_affine(
        self, other: "AffineOperator", binary_op: Callable[[Any, Any], Any]
    ) -> Self:
        return self._new(
            binary_op(self.operator.copy(), other.operator.copy()),
            binary_op(self.expression.copy(), other.expression.copy()),
        )

    def _combine_operator(
        self, other: Operator, binary_op: Callable[[Any, Any], Any]
    ) -> Self:
        return self._new(
            binary_op(self.operator.copy(), other.copy()), self.expression.copy()
        )

    def _combine_expression(
        self, other: Expression, binary_op: Callable[[Any, Any], Any]
    ) -> Self:
        return self._new(self.operator.copy(), binary_op(self.expression.copy(), other))

    def _combine_constlike(
        self,
        other: ScalarExpression | float | np.ndarray,
        binary_op: Callable[[Any, Any], Any],
    ):
        return self._new(
            binary_op(self.operator.copy(), other),
            binary_op(self.expression.copy(), other),
        )

    def _combine(self, other: Any, optype: BinaryOpType) -> Self:
        binary_op = BINARY_OPS[optype]
        if isinstance(other, type(self)):
            return self._combine_affine(other, binary_op)
        if isinstance(other, Operator):
            return self._combine_operator(other, binary_op)
        if isinstance(other, (ScalarExpression, float, np.ndarray)):
            if optype in [BinaryOpType.MUL, BinaryOpType.DIV]:
                return self._combine_constlike(other, binary_op)
            return self._combine_expression(other, binary_op)
        if isinstance(other, Expression):
            if optype in [BinaryOpType.MUL, BinaryOpType.DIV]:
                return self._combine_constlike(other, binary_op)
            return self._combine_expression(other, binary_op)
        return NotImplemented

    def __neg__(self):
        return self._new(-self.operator.copy(), -self.expression.copy())

    def __add__(self, other: Operator | Expression | float | np.ndarray) -> Self:
        return self._combine(other, BinaryOpType.ADD)

    def __radd__(self, other: Operator | Expression | float | np.ndarray) -> Self:
        return self._combine(other, BinaryOpType.ADD)

    def __sub__(self, other: Operator | Expression | float | np.ndarray) -> Self:
        return self._combine(other, BinaryOpType.SUB)

    def __rsub__(self, other: Operator | Expression | float | np.ndarray) -> Self:
        return (-self) + other

    def __mul__(self, other: Operator | Expression | float | np.ndarray) -> Self:
        return self._combine(other, BinaryOpType.MUL)

    def __rmul__(self, other: Operator | Expression | float | np.ndarray) -> Self:
        return self._combine(other, BinaryOpType.MUL)

    def __truediv__(self, other: Operator | Expression | float | np.ndarray) -> Self:
        return self._combine(other, BinaryOpType.DIV)
