from __future__ import annotations
from enum import IntEnum, auto
from dataclasses import dataclass


@dataclass(frozen=True)
class Space:
    shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)


class ShapeTransform(IntEnum):
    NONE = auto()
    REDUCE_RANK = auto()
    INCREASE_RANK = auto()


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
