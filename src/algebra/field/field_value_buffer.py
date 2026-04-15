from tools.buffer import ValueBuffer
from algebra.space import FieldShaped, TSpace
from algebra.exceptions import ShapeMismatchError

class FieldValueBuffer(FieldShaped[TSpace]):
    def __init__(
        self,
        space: TSpace,
        components: int,
        values: ValueBuffer
    ):
        if values.shape != (components, *space.shape):
            raise ShapeMismatchError((
                "To create FieldValueBuffer passed values has to match ",
                f"components x space shape. values.shape: {values.shape}",
                f" fieldshaped.shape: {(components, *space.shape)}"
            ))
        super().__init__(space, components)
        self._values = values

    @property
    def values(self) -> ValueBuffer:
        return self._values

