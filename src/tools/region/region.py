from typing import Union
from .region_utils import interior, boundary, full


class ShiftOutsideBounds(Exception):
    pass


class Region(tuple):
    """
    Wrapper around tuple[slice] for convenient sliced access into numpy arrays.
    Allows access to arrays using standard __getitem__ arr[region]. To access
    2dim array a region with tuple of two slices is required. Implements a
    series of helper methods like shift, extend, trim of the slice range.
    """

    def __new__(cls, slices: tuple[slice, ...]):
        return super().__new__(cls, slices)

    def __repr__(self) -> str:
        return f"Region({tuple(self)})"

    @property
    def ndim(self) -> int:
        return len(self)

    def shift(self, ax: int, offset: int) -> "Region":
        """Shift one axis by offset."""
        slices = list(self)
        s: slice = slices[ax]

        if offset == 0:
            return self

        if s.start is None and s.stop is None:
            return self

        if s.start is not None and s.stop is not None and s.start == s.stop:
            return self

        # compute new start
        if s.start is None:
            if offset < 0:
                raise ShiftOutsideBounds("cannot shift region beyond right bound")
            new_start = offset
        else:
            new_start = s.start + offset
            if s.start >= 0 and new_start < 0:
                raise ShiftOutsideBounds("cannot shift start past zero")

        # compute new stop
        if s.stop is None:
            if offset > 0:
                raise ShiftOutsideBounds("cannot shift region beyond left bound")
            new_stop = offset
        else:
            new_stop = s.stop + offset
            # crossing from negative up past zero is out of bounds; landing
            # on zero means the new stop is the array end (None)
            if s.stop < 0:
                if new_stop > 0:
                    raise ShiftOutsideBounds("cannot shift stop past zero")
                if new_stop == 0:
                    new_stop = None

        slices[ax] = slice(new_start, new_stop, s.step)
        return Region(tuple(slices))

    def extend(self, ax: int, offset: int) -> "Region":
        """
        Returns region with slice on axis ax extended by offset.
        Positive offset extends stop while negative extends slice start.
        Extend is clamped to 0 on start and has no effect if stop is None.
        Example:
            region(slice(2,3)).extend(0, 2) -> region(slice(2,5))
            region(slice(2,3)).extend(0, -2) -> region(slice(0,5))
            region(slice(0,None)).extend(0, -2) -> region(slice(0,None))
        """
        slices = list(self)
        s: slice = slices[ax]
        if offset == 0:
            return self
        if s.start is None and s.stop is None:
            return self
        is_empty = s.start is not None and s.stop is not None and s.start == s.stop
        if offset < 0:
            # extend start backward; negative offset on empty has no effect
            if is_empty or s.start is None:
                return self
            new_start = s.start + offset  # offset is negative
            # only clamp to None when start was non-negative (concrete index)
            # and crossed zero — for negative starts we keep them more negative
            if s.start >= 0 and new_start <= 0:
                new_start = None
            slices[ax] = slice(new_start, s.stop, s.step)
        else:
            if s.stop is None:
                return self
            new_stop = s.stop + offset
            # if old stop was negative and we crossed past zero, clamp to None
            if s.stop < 0 and new_stop >= 0:
                new_stop = None
            slices[ax] = slice(s.start, new_stop, s.step)
        return Region(tuple(slices))

    def trim(self, ax: int, offset: int) -> "Region":
        """
        Returns region with slice on axis ax trimmed by offset.
        Positive offset trims stop while negative trims slice start.
        If trim results in slice start >= stop resulting region will have empty
        slice on ax slice(0,0).
        Example:
            region(slice(2,3)).trim(0, 2) -> region(slice(2,2))
            region(slice(0,None)).trim(0, 2) -> region(slice(0,-2))
            region(slice(2,3)).trim(0, -2) -> region(slice(3,3))
        """
        slices = list(self)
        s: slice = slices[ax]
        if s.start is not None and s.stop is not None and s.start == s.stop:
            return self
        if offset == 0:
            return self
        if offset < 0:
            # trim start (move it forward by |offset|)
            if s.start is None:
                new_start = -offset  # offset < 0, this is positive
            else:
                new_start = s.start - offset  # subtract negative = add
            # only collapse to empty when both indices are comparable
            # (same sign region); collapse to slice(stop, stop)
            if s.stop is not None and self._collapsed(new_start, s.stop):
                slices[ax] = slice(s.stop, s.stop, s.step)
                return Region(tuple(slices))
            slices[ax] = slice(new_start, s.stop, s.step)
        else:
            # trim stop (move it backward by offset)
            if s.stop is None:
                new_stop = -offset
            else:
                new_stop = s.stop - offset
            if s.start is not None and self._collapsed(s.start, new_stop):
                slices[ax] = slice(s.start, s.start, s.step)
                return Region(tuple(slices))
            slices[ax] = slice(s.start, new_stop, s.step)
        return Region(tuple(slices))

    @staticmethod
    def _collapsed(start: int, stop: int) -> bool:
        """Return True if start >= stop and the values are comparable.

        Mixed-sign indices (e.g. positive start with negative stop) cannot be
        compared without an array length, so they are treated as not collapsed.
        """
        if start >= 0 and stop >= 0:
            return start >= stop
        if start < 0 and stop < 0:
            return start >= stop
        return False

    def normalize(self, shape: tuple[int, ...]):
        """Clamp end of each slice in region to max(shape[ax])"""
        assert self.ndim == len(shape), "shape dims have to match ndim"
        slices = list[slice]()
        for s, ax_len in zip(self, shape):
            if s.start is None:
                start = 0
            else:
                start = s.start
            if s.stop is None:
                stop = ax_len
            else:
                stop = s.stop
            slices.append(slice(min(max(0, start), ax_len), min(ax_len, stop), s.step))
        return Region(tuple(slices))

    def replace(self, axis: int, new_slice: slice) -> "Region":
        """Return region with slice replaced one axis slice."""
        slices = list(self)
        slices[axis] = new_slice
        return Region(slices)

    def intersect(self, other: "Region") -> "Region":
        assert self.ndim == other.ndim, "Can only intersect regions of equal ndim"
        for ax in range(self.ndim):
            if not self[ax].step == other[ax].step == None:
                raise ValueError("intersect only works on slices with None step")

        slices = list[slice]()
        for s1, s2 in zip(self, other):
            start = max(
                s1.start if s1.start is not None else float("-inf"),
                s2.start if s2.start is not None else float("-inf"),
            )
            stop = min(
                s1.stop if s1.stop is not None else float("inf"),
                s2.stop if s2.stop is not None else float("inf"),
            )

            if start >= stop:
                slices.append(slice(0, 0))
            else:
                slices.append(
                    slice(
                        None if start == float("-inf") else int(start),
                        None if stop == float("inf") else int(stop),
                        None,
                    )
                )

        return Region(tuple(slices))

    @staticmethod
    def full(ndim: int) -> "Region":
        return full(ndim)

    @staticmethod
    def interior(
        ndim: int, offsets: tuple[Union[tuple[int, int], int], ...]
    ) -> "Region":
        return interior(ndim, offsets)

    @staticmethod
    def boundary(ndim: int, ax: int, side: int) -> "Region":
        return boundary(ndim, ax, side)
