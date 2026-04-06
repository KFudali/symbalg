import numpy as np

from ..region import Region, interior, boundary_region


class StructuredGridND:
    def __init__(
        self,
        shape: tuple[int, ...],
        spacing: tuple[float, ...]
    ):
        self._shape = tuple(shape)
        self._spacing = tuple(spacing)
        self._ndim = len(self._shape)
        assert len(self._shape) == len(self._spacing)

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
    def spacing(self) -> tuple[int, ...]:
        return self._spacing

    @property
    def interior(self) -> Region:
        return interior(self.shape, tuple(np.ones(self.ndim, dtype=int)))

    def points(self) -> np.ndarray:
        axes = [
            np.arange(self.shape[d]) * self.spacing[d]
            for d in range(self.ndim)
        ]
        return np.meshgrid(*axes, indexing="ij")

    def boundary(self, ax: int, side: int) -> Region:
        if side not in (-1, 1):
            raise ValueError("side must be -1 (left boundary) or 1 (right boundary)")
        if not 0 <= ax < self.ndim:
            raise IndexError(f"ax must be in [0, {self.ndim-1}]; got {ax}")
        include_corners = ax % 2 == 0
        return boundary_region(self.shape, ax, side, include_corners)

    def ax_spacing(self, axis: int) -> float:
        return self._spacing[axis]

    def flat_id(self, idx) -> int:
        """Convert ND index → flat index"""
        return np.ravel_multi_index(idx, self.shape)

    def flat_id_arr(self, idx_arr: np.ndarray) -> np.ndarray:
        """
        Convert ND index array → flat index array
        idx_arr shape: (N, ndim)
        returns shape: (N,)
        """
        idx_arr = np.asarray(idx_arr)
        if idx_arr.ndim != 2 or idx_arr.shape[1] != len(self.shape):
            raise ValueError(
                f"Expected idx_arr of shape (N, {len(self.shape)}), "
                f"got {idx_arr.shape}"
            )
        return np.ravel_multi_index(idx_arr.T, self.shape)

    def idx(self, flat_id: int) -> tuple[int, ...]:
        """Convert flat index → ND index"""
        return np.unravel_index(flat_id, self.shape)

    def idx_arr(self, flat_ids: np.ndarray) -> np.ndarray:
        """Convert flat index array → rray of dim x ids"""
        return np.stack((np.unravel_index(flat_ids, self.shape)), axis=1)

    def offset_flat_id(self, flat_id: int, offset: tuple[int, ...]) -> int:
        idx = np.array(self.idx(flat_id))
        offset = np.array(offset)

        new_idx = idx - offset
        new_idx = np.clip(new_idx, 0, np.array(self.shape) - 1)

        return self.flat_id(*new_idx)
