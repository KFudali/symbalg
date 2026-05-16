from tools.buffer import ValueBuffer
from algebra.core.space import FieldShaped, FieldShape
from algebra.exceptions import ShapeMismatchError


class FieldValueBuffer(FieldShaped):
    def __init__(self, fieldshape: FieldShape, value_buffer: ValueBuffer):
        if value_buffer.shape != fieldshape.shape:
            raise ShapeMismatchError(
                (
                    "To create FieldValueBuffer passed ValueBuffer has to match ",
                    f"fieldshape shape. value_buffer.shape: {value_buffer.shape}",
                    f" fieldshape.shape: {(fieldshape.shape)}",
                )
            )
        super().__init__(fieldshape)
        self._value_buffer = value_buffer

    @property
    def values(self) -> ValueBuffer:
        return self._value_buffer
