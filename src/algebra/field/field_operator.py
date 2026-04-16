from __future__ import annotations
from typing import Self, Any, Callable, Generic
import numpy as np

from algebra.core.operator import Operator
from algebra.core.expression import Expression, CallableExpression, ScalarExpression
from algebra.core.space import TSpace

from algebra.exceptions import ShapeMismatchError
from algebra.symbolic import SymbolicExpression, AffineOperator
from tools.symbolic import BinaryOpType, BINARY_OPS

from .field import Field

class FieldOperator(AffineOperator, Generic[TSpace]):
    def __init__(
        self,
        field: Field[TSpace],
        operator: Operator,
        expression: Expression | None = None
    ):
        if field.shape != operator.input_shape:
            raise ShapeMismatchError("Field has to match operator.")
        self._field = field
        super().__init__(operator, expression)

    @property
    def field(self) -> Field[TSpace]:
        return self._field

    def _new(self, op: Operator, exp: Expression) -> "FieldOperator":
        return FieldOperator(self._field, op, exp)

    def _combine(
        self,
        other: Any,
        optype: BinaryOpType
    ) -> Self | SymbolicExpression:
        binary_op = BINARY_OPS[optype]
        if isinstance(other, type(self)):
            if other.field == self.field:
                return self._combine_affine(other, binary_op)
            self_eval = CallableExpression(self.eval, self.output_shape)
            other_eval = CallableExpression(other.eval, other.output_shape)
            result = SymbolicExpression(self_eval)
            return result + other_eval
        if isinstance(other, Operator):
            # Cannot combine with any Operator, they have to share input fields.
            # The only way to modify interior operator is to combine self with other
            # object of FieldOperator type
            return NotImplemented
        if isinstance(other, (ScalarExpression, float, np.ndarray)):
            if optype in [BinaryOpType.MUL, BinaryOpType.DIV]:
                return self._combine_constlike(other, binary_op)
            return self._combine_expression(other, binary_op)
        if isinstance(other, Expression):
            if optype in [BinaryOpType.MUL, BinaryOpType.DIV]:
                return self._combine_constlike(other, binary_op)
            return self._combine_expression(other, binary_op)
        return NotImplemented

    def eval(self) -> np.ndarray:
        output_field = np.zeros(shape=self.output_shape, dtype=float)
        input_field = self._field.value().eval()
        self.operator.apply(input_field, output_field)
        output_field += self.expression.eval()
        return output_field

    def copy(self) -> Self:
        return FieldOperator(
            self._field, self.operator.copy(), self.expression.copy()
        )

    def __neg__(self) -> Self:
        return self._new(-self.operator.copy(), -self.expression.copy())

    def __add__(
        self,
        other: "FieldOperator" | Expression | float | np.ndarray
    ) -> Self | SymbolicExpression:
        return self._combine(other, BinaryOpType.ADD)

    def __radd__(
        self,
        other: "FieldOperator" | Expression | float | np.ndarray
    ) -> Self | SymbolicExpression:
        return self._combine(other, BinaryOpType.ADD)

    def __sub__(
        self,
        other: "FieldOperator" | Expression | float | np.ndarray
    ) -> Self | SymbolicExpression:
        return self._combine(other, BinaryOpType.SUB)

    def __rsub__(
        self,
        other: "FieldOperator" | Expression | float | np.ndarray
    ) -> Self | SymbolicExpression:
        return (-self) + other

    def __mul__(
        self,
        other: "FieldOperator" | Expression | float | np.ndarray
    ) -> Self | SymbolicExpression:
        return self._combine(other, BinaryOpType.MUL)

    def __rmul__(
        self,
        other: "FieldOperator" | Expression | float | np.ndarray
    ) -> Self | SymbolicExpression:
        return self._combine(other, BinaryOpType.MUL)

    def __truediv__(
        self, other:
        "FieldOperator" | Expression | float | np.ndarray
    ) -> Self | SymbolicExpression:
        return self._combine(other, BinaryOpType.DIV)
