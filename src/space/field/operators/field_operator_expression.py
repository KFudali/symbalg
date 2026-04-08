from typing import Self
import numpy as np
from algebra.operator import SymbolicOperator, Operator
from algebra.expression import SymbolicExpression, Expression, ZeroExpression, CallableExpression
from algebra.exceptions import ShapeMismatchError
from space.field.core import AbstractField

class FieldOperatorExpression(Operator, Expression):
    def __init__(
        self,
        field: AbstractField,
        operator: Operator,
        lifting: Expression | None = None
    ):
        if field.shape != operator.input_shape:
            raise ShapeMismatchError("Field has to match operator.")
        if not lifting:
            lifting = ZeroExpression(operator.input_shape)
        self._operator = SymbolicOperator(operator)
        self._lifting = SymbolicExpression(lifting)
        self._field = field
        super().__init__(operator.input_shape, operator.output_shape)

    @property
    def field(self) -> AbstractField:
        return self._field

    @property
    def operator(self) -> SymbolicOperator:
        return self._operator

    @property
    def lifting(self) -> SymbolicExpression:
        return self._lifting

    def eval(self) -> np.ndarray:
        output_field = np.zeros(shape=self.output_shape, dtype=float)
        input_field = self._field.get_current()
        self._operator.apply(input_field, output_field)
        output_field += self._lifting.eval()
        return output_field

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        self._operator.apply(input_field, output_field)
        output_field += self._lifting.eval()

    def copy(self) -> Self:
        return FieldOperatorExpression(
            self._field, self._operator.copy(), self._lifting.copy()
        )

    def __add__(self, other: Operator | Expression | float):
        if isinstance(other, FieldOperatorExpression):
            if self.field == other.field:
                return FieldOperatorExpression(
                    self._field,
                    self.operator.copy() + other.operator.copy(),
                    self.lifting.copy() + other.lifting.copy()
                )
            else:
                eval_self = CallableExpression(self.copy().eval, self.output_shape)
                eval_other = CallableExpression(other.copy().eval, other.output_shape)
                eval_symbolic = SymbolicExpression(eval_self)
                return eval_symbolic + eval_other
        if isinstance(other, Operator):
            return FieldOperatorExpression(
                self._field, self.operator.copy() + other, self.lifting.copy()
            )
        if isinstance(other, (Expression, float, np.ndarray)):
            eval_self = CallableExpression(self.copy().eval, self.output_shape)
            eval_symbolic = SymbolicExpression(eval_self)
            return eval_symbolic + other
