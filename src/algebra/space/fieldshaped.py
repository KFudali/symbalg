from dataclasses import dataclass
from .space import Space


@dataclass(frozen=True)
class FieldShape:
    space: Space
    components: tuple[int, ...]

    def is_scalar(self) -> bool:
        return len(self.components) == 0

    @classmethod
    def scalar(cls, space: Space) -> "FieldShape":
        return cls(space, ())

    @property
    def shape(self) -> tuple[int, ...]:
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
