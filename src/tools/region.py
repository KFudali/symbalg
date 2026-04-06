from typing import Union
import numpy as np


class Region(tuple):
    def __new__(cls, slices: list[slice]):
        slices = tuple(slices)
        if not all(isinstance(s, slice) for s in slices):
            raise TypeError("Region must contain only slice objects")
        return super().__new__(cls, slices)

    @classmethod
    def full(cls, ndim: int) -> "Region":
        """Create full region (:, :, ..., :)"""
        return cls(slice(None) for _ in range(ndim))

    @property
    def ndim(self) -> int:
        return len(self)

    def shift(self, axis: int, offset: int) -> "Region":
        """Shift one axis by offset."""
        slices = list(self)
        s: slice = slices[axis]

        slices[axis] = slice(
            None if s.start is None else s.start + offset,
            None if s.stop is None else s.stop + offset,
            s.step,
        )

        return Region(slices)

    def replace(self, axis: int, new_slice: slice) -> "Region":
        """Replace one axis slice."""
        slices = list(self)
        slices[axis] = new_slice
        return Region(slices)

    def __repr__(self) -> str:
        return f"Region({tuple(self)})"


def interior(
    shape: tuple[int, ...],
    offsets: tuple[
        Union[int, tuple[int, int]], ...
    ],
) -> Region:
    """
    Compute interior slice for an array of given shape.

    offsets per axis can be:
        int                  -> symmetric (-k, +k)
        (left, right) tuple  -> asymmetric trim

    Example:
        shape = (10, 20)

        offsets = (2, 3)
        -> trims 2 on axis0, 3 on axis1

        offsets = ((1, 2), (0, 4))
        -> axis0 trims 1 left, 2 right
           axis1 trims 0 left, 4 right
    """
    if len(shape) != len(offsets):
        raise ValueError("shape and offsets must have same length")

    region: list[slice] = []

    for ax, (dim, off) in enumerate(zip(shape, offsets)):
        if isinstance(off, (int, np.integer)):
            left = right = abs(off)
        else:
            left, right = off
            left = abs(left)
            right = abs(right)

        if left + right > dim:
            raise ValueError(
                f"Offsets too large for axis {ax}: "
                f"{left} + {right} > {dim}"
            )

        start = left
        stop = dim - right

        region.append(slice(start, stop))

    return Region(region)


def boundary_region(
    shape: tuple[int, ...], ax: int, side: int, include_corners: bool = True
) -> Region:
    """Return the boundary region along axis ``ax``.

        - ``side``: -1 -> left side (index 0), 1 -> right side (last index)
        - If ``include_corners`` is False, slices on other axes are trimmed to
            exclude the extreme indices (i.e. exclude corners/edges). Trimming is
            applied only when the corresponding dimension is > 2 to avoid empty
            ranges for very small dimensions.
    """
    ndim = len(shape)

    if not isinstance(ax, int):
        raise TypeError("ax must be an integer axis index")
    if not (-ndim <= ax < ndim):
        raise IndexError(f"ax must be in [-{ndim}, {ndim-1}]; got {ax}")
    if side not in (-1, 1):
        raise ValueError("inward_side must be -1 (low/left) or 1 (high/right)")

    if ax < 0:
        ax += ndim

    region: list[slice] = [slice(0, shape[i]) for i in range(ndim)]

    if side == -1:
        region[ax] = slice(0, 1)
    else:
        region[ax] = slice(shape[ax] - 1, shape[ax])

    if not include_corners:
        for axis in range(ndim):
            if axis == ax:
                continue
            dim = shape[axis]
            if dim > 2:
                region[axis] = slice(1, dim - 1)

    return Region(region)
