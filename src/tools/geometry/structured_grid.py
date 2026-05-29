import numpy as np
from tools.region import Region, interior, boundary


class StructuredGridND:
    def __init__(self, shape: tuple[int, ...], spacing: tuple[float, ...]):
        assert len(shape) == len(spacing)
        self._shape = tuple(shape)
        self._spacing = tuple(spacing)
        self._ndim = len(self._shape)

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def spacing(self) -> tuple[float, ...]:
        return self._spacing

    @property
    def interior(self) -> Region:
        return interior(self.ndim, tuple(1 for _ in range(self.ndim)))

    def points(self) -> tuple[np.ndarray, ...]:
        axes = [np.arange(self.shape[d]) * self.spacing[d] for d in range(self.ndim)]
        return np.meshgrid(*axes, indexing="ij")

    def boundary(self, ax: int, side: int) -> Region:
        if side not in (-1, 1):
            raise ValueError("side must be -1 (left boundary) or 1 (right boundary)")
        if not 0 <= ax < self.ndim:
            raise IndexError(f"ax must be in [0, {self.ndim-1}]; got {ax}")
        exclude_corners = ax % 2 != 0
        return boundary(self.ndim, ax, side, exclude_corners)

    def ax_spacing(self, axis: int) -> float:
        return self._spacing[axis]
