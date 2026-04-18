from algebra.core.expression import Expression, CallableExpression
from algebra.core.space import FieldShaped, TSpace

from algebra.exceptions import ShapeMismatchError
from algebra.symbolic import SymbolicExpression

from tools.buffer import ValueBuffer, ShiftProxyValueBuffer
from tools.action import LazyAction

from .field_value_buffer import FieldValueBuffer

class Field(FieldShaped[TSpace]):
    def __init__(self, value_buffer: FieldValueBuffer):
        super().__init__(value_buffer.space, value_buffer.components)
        self._buffer = value_buffer

    @property
    def buffer(self) -> ValueBuffer:
        return self._buffer.values

    def past(self, step: int = 1) -> "Field":
        self.buffer.set_saved_steps(step)
        buffer = ShiftProxyValueBuffer(self.buffer, step)
        field_buffer = FieldValueBuffer(self.space, self.components, buffer)
        return Field(field_buffer)

    def value(self) -> SymbolicExpression:
        getter = CallableExpression(self._buffer.values.get, self._buffer.shape)
        return SymbolicExpression(getter)

    def advance(self):
        self.buffer.advance()

    def reset(self):
        self.buffer.reset()

    def set_value(self, value: Expression) -> LazyAction:
        if value.output_shape != self.shape:
            raise ShapeMismatchError((
                f"Cannot set_value, passed value shape: {value.output_shape} does not ",
                f"match field shape: {self.shape}"
            ))
        def set_buffer():
            self.buffer.set(value.eval())
        return LazyAction(set_buffer)
