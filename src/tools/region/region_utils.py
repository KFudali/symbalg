from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .region import Region


def full(ndim: int) -> Region:
    """Create full region (:, :, ..., :)"""
    from .region import Region  # local import avoids circular import at module load

    return Region(slice(None) for _ in range(ndim))


def interior(ndim: int, offsets: tuple[Union[tuple[int, int], int], ...]) -> Region:
    """
    Compute interior slice for ndim
    Parameters:
        ndim (int)
        offsets (tuple[tuple[int, int] | int]):
            offsets from left-right edge on each ax if int treated as left=right
    Example:
        ndim = 2
        offsets = ((1,2), 2)
        -> axis0 trims 1 left, 2 right
           axis1 trims 2 left, 2 right
        offsets = ((1, 2), (0, 4))
        -> axis0 trims 1 left, 2 right
           axis1 trims 0 left, 4 right
    """
    from .region import Region

    if ndim != len(offsets):
        raise ValueError("ndim has to match offsets length")

    region = list[slice]()
    for off in offsets:
        if isinstance(off, (int,)):
            left = right = off
        else:
            left, right = off

        if left < 0 or right < 0:
            raise ValueError("offsets must be non-negative")

        start = left if left != 0 else None
        stop = -right if right != 0 else None
        region.append(slice(start, stop))

    return Region(region)


def boundary(ndim: int, ax: int, side: int, exclude_corners: bool = False) -> Region:
    """Return the boundary region along axis ``ax``.

    Parameters:
        shape (tuple[int, ...]): needed for right boundary
        side (int -1, 1): -1 -> left (index 0), 1 -> right (last index)
    """
    from .region import Region

    if not -ndim <= ax < ndim:
        raise IndexError(f"ax must be in [-{ndim}, {ndim-1}]; got {ax}")
    if side not in (-1, 1):
        raise ValueError("inward_side must be -1 (low/left) or 1 (high/right)")

    region: list[slice] = [slice(None, None) for i in range(ndim)]
    if side == -1:
        region[ax] = slice(None, 1)
    else:
        region[ax] = slice(-1, None)
    reg = Region(region)
    if exclude_corners:
        for axs in range(ndim):
            if axs == ax:
                continue
            reg = reg.trim(axs, 1)
            reg = reg.trim(axs, -1)
    return reg
