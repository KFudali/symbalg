from __future__ import annotations
from typing import Callable, Any
import numpy as np

from algebra.operator import Operator
from algebra.expression import Expression
from .symbolic_expression import SymbolicExpression
from .symbolic_operator import SymbolicOperator

class AffineOperator(Operator):
    def __init__(
        self,
        operator: Operator,
        expression: Expression,
    ):
        self._operator = SymbolicOperator(operator)
        self._expression = SymbolicExpression(expression)
        super().__init__(operator.input_shape, operator.output_shape)

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        self.operator.apply(input_field, output_field)
        output_field += self.expression.eval()

    def _new(self, op: Operator, exp: Expression):
        pass

    @property
    def operator(self) -> Operator:
        return self._operator

    @property
    def expression(self) -> Expression:
        return self._expression

    def _combine_affine(
        self, other: "AffineOperator", binary_op: Callable[[Any, Any], Any]
    ):
        return self._new(
            binary_op(self.operator.copy(), other.operator.copy()),
            binary_op(self.expression.copy(), other.expression.copy()),
        )

    def _combine_operator(
        self, other: Operator, binary_op: Callable[[Any, Any], Any]
    ):
        return self._new(
            binary_op(self.operator.copy(), other.operator.copy()),
            self.expression.copy()
        )

    def _combine_expression(
        self, other: Expression,
        binary_op: Callable[[Any, Any], Any]
    ):
        return self._new(
            self.operator.copy(),
            binary_op(self.expression.copy() + other)
        )

    def _combine_constlike(
        self, other: float | np.ndarray,
        binary_op: Callable[[Any, Any], Any]
    ):
        return self._new(
            binary_op(self.operator.copy(), other),
            binary_op(self.expression.copy(), other)
        )

    def _combine(self, other: Any, binary_op: Callable[[Any, Any], Any]):
        if isinstance(other, type(self)):
            return self._combine_affine(other, binary_op)
        if isinstance(other, Operator):
            return self._combine_operator(other, binary_op)
        if isinstance(other, Expression):
            return self._combine_expression(other, binary_op)
        if isinstance(other, (float, np.ndarray)):
            return self._combine_constlike(other,binary_op)
        return NotImplemented

    def __add__(self, other: Operator | Expression | float | np.ndarray):
        return self._combine(other, lambda a, b: a + b)

    def __radd__(self, other: Operator | Expression | float | np.ndarray):
        return self._combine(other, lambda a, b: a + b)

    def __sub__(self, other: Operator | Expression | float | np.ndarray):
        return self._combine(other, lambda a, b: a - b)

    def __rsub__(self, other: Operator | Expression | float | np.ndarray):
        return self._combine(other, lambda a, b: (-a) + b)

    def __mul__(self, other: Operator | Expression | float | np.ndarray):
        return self._combine(other, lambda a, b: a * b)

    def __rmul__(self, other: Operator | Expression | float | np.ndarray):
        return self._combine(other, lambda a, b: a * b)

    def __truediv__(self, other: Operator | Expression | float | np.ndarray):
        return self._combine(other, lambda a, b: a / b)
