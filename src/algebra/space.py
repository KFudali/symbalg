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

    def transform(
        self, space: "Space", shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        assert len(shape) >= space.ndim, (
            "Input shape must include the spatial dimensions"
        )
        split = len(shape) - space.ndim
        components, spatial = shape[:split], shape[split:]
        if self is ShapeTransform.NONE:
            return shape
        if self is ShapeTransform.INCREASE_RANK:
            return (*components, space.ndim, *spatial)
        if self is ShapeTransform.REDUCE_RANK:
            assert len(components) >= 1, (
                "Cannot reduce rank of a field with no component axes"
            )
            assert components[-1] == space.ndim, (
                "Leading component axis must match space.ndim to reduce rank"
            )
            return (*components[:-1], *spatial)
        raise ValueError(f"Unknown ShapeTransform: {self}")

    def reverse(
        self, space: "Space", shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        assert len(shape) >= space.ndim, (
            "Output shape must include the spatial dimensions"
        )
        split = len(shape) - space.ndim
        components, spatial = shape[:split], shape[split:]
        if self is ShapeTransform.NONE:
            return shape
        if self is ShapeTransform.INCREASE_RANK:
            assert len(components) >= 1, (
                "Cannot reverse INCREASE_RANK: missing inserted component axis"
            )
            assert components[-1] == space.ndim, (
                "Inserted component axis must match space.ndim"
            )
            return (*components[:-1], *spatial)
        if self is ShapeTransform.REDUCE_RANK:
            return (*components, space.ndim, *spatial)
        raise ValueError(f"Unknown ShapeTransform: {self}")


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
