from __future__ import annotations
from tools.buffer import ValueBuffer, ShiftProxyValueBuffer
from .expression import Expression, CallableExpression
from .space import FieldShaped, FieldShape


class Field(FieldShaped):
    def __init__(self, shape: FieldShape, value_buffer: ValueBuffer):
        assert shape.shape == value_buffer.shape
        super().__init__(shape)
        self._value_buffer = value_buffer

    def past(self, step: int) -> "Field":
        return Field(self.fieldshape, ShiftProxyValueBuffer(self._value_buffer, step))

    def value(self) -> Expression:
        return CallableExpression(self.shape, self._value_buffer.get)
