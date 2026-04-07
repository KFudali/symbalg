import numpy as np
from tools.buffer import ValueBuffer, ShiftProxyValueBuffer

import algebra
from space.core import Space
from .core import AbstractField, FieldValue, FieldUpdate

class FieldView(AbstractField):
    def __init__(
        self, 
        space: Space, components: int,
        value_buffer: ValueBuffer
    ):
        super().__init__(space, components)
        if self.shape != value_buffer.shape:
            raise algebra.exceptions.ShapeMismatchError(
                "Passed value buffer does not match field shape"
            )
        self._value_buffer = value_buffer

    def value(self) -> FieldValue:
        return FieldValue(self)

    def save_past(self, steps: int):
        self._value_buffer.set_saved_steps(steps)

    def past(self, steps: int = 1) -> "FieldView":
        if steps > self._value_buffer.saved_steps: self.save_past(steps)
        return FieldView(
            self.space, 
            self.components,
            ShiftProxyValueBuffer(self._value_buffer, steps), 
        )

    def advance(self, dt: float):
        self._value_buffer.advance()

    def get_current(self):
        return self._value_buffer.get(0)
    
    def set_current(self, value: np.ndarray):
        return self._value_buffer.set(value)


class Field(FieldView):
    def __init__(
        self, 
        space: Space, components: int,
        value_buffer: ValueBuffer,
    ):
        super().__init__(space, components, value_buffer)

    def update(self, expr: algebra.Expression) -> FieldUpdate:
        return FieldUpdate(self, expr)
