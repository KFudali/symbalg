from dataclasses import dataclass
from typing import Self

import math
from algebra.exceptions import ShapeMismatchError
from .space import Space


@dataclass(frozen=True)
class Shape:
    space: Space
    components: tuple[int, ...] = ()

    @classmethod
    def from_array(cls, space: Space, array_shape: tuple[int, ...]) -> Self:
        assert len(array_shape) >= space.ndim
        return cls(space, array_shape[: -space.ndim])

    @classmethod
    def from_sparse(cls, space: Space, sparse_shape: tuple[int, ...]) -> Self:
        n = math.prod(space.shape)
        if len(sparse_shape) < 2 or not (sparse_shape[-1] == sparse_shape[-2] == n):
            raise ShapeMismatchError(
                f"Sparse shape {sparse_shape} not compatible with space {space}"
            )
        return cls(space, sparse_shape[:-2])

    @classmethod
    def scalar(cls, space: Space) -> Self:
        return cls(space, (-1,))

    def is_scalar(self) -> bool:
        return len(self.components) == 1 and self.components[0] == -1

    def fieldshape(self) -> tuple[int, ...]:
        if self.is_scalar():
            return ()
        return (*self.components, *self.space.shape)

    def sparseshape(self) -> tuple[int, ...]:
        return (*self.components, self.n, self.n)

    @property
    def n(self) -> int:
        return math.prod(self.space.shape)


class SpaceShaped:
    def __init__(self, shape: Shape):
        self._shape = shape

    @property
    def components(self) -> tuple[int, ...]:
        return self._shape.components

    @property
    def space(self) -> Space:
        return self._shape.space

    @property
    def shape(self) -> Shape:
        return self._shape
