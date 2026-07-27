import numpy as np
from dataclasses import dataclass

from .space import Space


@dataclass(frozen=True)
class SparseShape:
    """Stores shape of array of coefficients for each node of space.

    Described by components and space, returns shape as
    (*components, n, n)
    where n is total number of points in space (np.prod(space.shape))
    """

    space: Space
    components: tuple[int, ...]

    @classmethod
    def from_shape(cls, space: Space, shape: tuple[int, ...]) -> "SparseShape":
        assert len(shape) >= space.ndim
        n = int(np.prod(space.shape))
        assert shape[-1] == shape[-2] == n
        return SparseShape(space, shape[:-2])

    @property
    def shape(self) -> tuple[int, ...]:
        n = int(np.prod(self.space.shape))
        return (*self.components, n, n)


class SparseShaped:
    def __init__(self, shape: SparseShape):
        self._shape = shape

    @property
    def comps(self) -> tuple[int, ...]:
        return self._shape.components

    @property
    def sparseshape(self) -> SparseShape:
        return self._shape

    @property
    def space(self) -> Space:
        return self._shape.space

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape.shape
