from dataclasses import dataclass
from .space import Space


@dataclass(frozen=True)
class FieldShape:
    space: Space
    components: tuple[int, ...]

    def is_scalar(self) -> bool:
        return len(self.components) == 1 and self.components[0] == -1

    @classmethod
    def scalar(cls, space: Space) -> "FieldShape":
        return cls(space, (-1,))

    @classmethod
    def from_shape(cls, space: Space, shape: tuple[int, ...]) -> "FieldShape":
        assert len(shape) >= space.ndim
        return FieldShape(space, shape[: -space.ndim])

    @property
    def shape(self) -> tuple[int, ...]:
        if self.is_scalar():
            return ()
        return (*self.components, *self.space.shape)


class FieldShaped:
    def __init__(self, shape: FieldShape):
        self._shape = shape

    @property
    def fieldshape(self) -> FieldShape:
        return self._shape

    @property
    def space(self) -> Space:
        return self._shape.space

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape.shape
