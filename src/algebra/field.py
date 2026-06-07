from __future__ import annotations
from tools.buffer import ValueBuffer, ShiftProxyValueBuffer
from tools.action import LazyAction
from .expression import Expression, CallableExpression
from .space import FieldShaped, FieldShape
from .symbolic import SymbolicExpression


class Field(FieldShaped):
    def __init__(self, shape: FieldShape, value_buffer: ValueBuffer):
        assert shape.shape == value_buffer.shape
        super().__init__(shape)
        self._value_buffer = value_buffer

    def past(self, step: int) -> "Field":
        return Field(self.fieldshape, ShiftProxyValueBuffer(self._value_buffer, step))

    def value(self) -> SymbolicExpression:
        return SymbolicExpression.wrap(
            CallableExpression(self.shape, self._value_buffer.get)
        )

    def set_value(self, value: Expression) -> LazyAction:
        assert value.shape == self.shape

        def set():
            self._value_buffer.set(value.eval())

        return LazyAction(set)
