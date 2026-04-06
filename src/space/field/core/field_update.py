from .abstract_field import AbstractField
from .fieldshaped import FieldShaped
from algebra.expression import Expression

class FieldUpdate(SimpleExpression, FieldShaped):
    def __init__(
        self, 
        field: AbstractField, 
        expr: Expression,
    ):
        super().__init__(None)
        self._field = field
        self._expr = expr

    def copy(self) -> "FieldUpdate":
        return FieldUpdate(self._field, self._expr.copy())

    def eval(self):
        self._field.set_current(self._expr.eval())
