from .abstract_field import AbstractField
from algebra.expression import Expression
from tools.action import LazyAction

class FieldUpdate(LazyAction):
    def __init__(
        self, 
        field: AbstractField, 
        expr: Expression,
    ):
        self._field = field
        self._expr = expr
        def set_field():
            self._field.set_current(self._expr.eval())
        super().__init__(set_field, None)

    def copy(self) -> "FieldUpdate":
        return FieldUpdate(self._field, self._expr.copy())

