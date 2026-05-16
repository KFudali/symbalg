from algebra.core.expression import Expression, CallableExpression
from algebra.fieldshape import FieldShaped

from algebra.exceptions import ShapeMismatchError
from algebra.symbolic import SymbolicExpression

from tools.buffer import ValueBuffer, ShiftProxyValueBuffer
from tools.action import LazyAction


class Field(FieldShaped):
    def __init__(self, shape: FieldShape, value_buffer: ValueBuffer):
        assert shape == value_buffer.shape, "Value buffer shape has to match shape"
        super().__init__(shape)
        self._value_buffer = value_buffer

    def past(self, step: int = 1) -> "Field":
        self._value_buffer.set_saved_steps(step + 1)
        buffer = ShiftProxyValueBuffer(self._value_buffer, step)
        return Field(self.shape, buffer)

    def value(self) -> SymbolicExpression:
        getter = CallableExpression(self._value_buffer.get, self.shape)
        return SymbolicExpression(getter)

    def advance(self):
        self._value_buffer.advance()

    def reset(self):
        self._value_buffer.reset()

    def set_value(self, value: Expression) -> LazyAction:
        if value.shape != self.shape:
            raise ShapeMismatchError(
                (
                    f"Cannot set_value, passed value shape: {value.shape} does not ",
                    f"match field shape: {self.shape}",
                )
            )

        def set_buffer():
            self._value_buffer.set(value.eval())

        return LazyAction(set_buffer, None)
