from dataclasses import dataclass
from typing import Callable
import numpy as np
from tools import region
from .stencil import Stencil


@dataclass(frozen=True)
class AxStencil:
    interior: Stencil
    lefts: tuple[Stencil, ...]
    rights: tuple[Stencil, ...]

    def copy(self) -> "AxStencil":
        return AxStencil(
            self.interior.copy(),
            tuple(dx.copy() for dx in self.lefts),
            tuple(dx.copy() for dx in self.rights),
        )

    def __post_init__(self):
        left_range, right_range = self.interior.max_offsets()
        assert left_range <= len(self.lefts)
        assert right_range <= len(self.rights)
        for dist_to_boundary, approx in enumerate(self.lefts):
            if approx.left_offsets():
                assert max(approx.left_offsets()) <= dist_to_boundary
        for dist_to_boundary, approx in enumerate(self.rights):
            if approx.right_offsets():
                assert max(approx.right_offsets()) <= dist_to_boundary

    def eval(self, ax: int, field: np.ndarray) -> np.ndarray:
        out = np.zeros_like(field)
        self.eval_to(ax, field, out)
        return out

    def _eval_to_boundary(self, ax: int, side: int, field: np.ndarray, out: np.ndarray):
        boundary = region.boundary(field.ndim, ax, side)
        if side == -1:
            stencils = self.lefts
        else:
            stencils = self.rights
        for dist_to_boundary, stencil in enumerate(stencils):
            left_offset, right_offset = stencil.max_offsets()
            out_reg = boundary.shift(ax, -side * dist_to_boundary)
            field_reg = out_reg.extend(ax, -left_offset).extend(ax, right_offset)
            stencil.eval_to(ax, field[field_reg], out[out_reg])

    def _eval_to_interior(self, ax: int, field: np.ndarray, out: np.ndarray):
        offsets = [(0, 0) for _ in range(field.ndim)]
        offsets[ax] = (len(self.lefts), len(self.rights))
        out_interior = region.interior(field.ndim, tuple(offsets))

        offsets = [(0, 0) for _ in range(field.ndim)]
        left, right = self.interior.max_offsets()
        left = len(self.lefts) - left
        right = len(self.rights) - right
        offsets[ax] = (left, right)
        field_interior = region.interior(field.ndim, tuple(offsets))
        self.interior.eval_to(ax, field[field_interior], out[out_interior])

    def eval_to(self, ax: int, field: np.ndarray, out: np.ndarray):
        assert field.shape == out.shape, "Can only eval_to fields that match shapes"
        self._eval_to_boundary(ax, -1, field, out)
        self._eval_to_boundary(ax, 1, field, out)
        self._eval_to_interior(ax, field, out)
    
    def norm_offset(
        first: tuple[Stencil, ...], append_first: Stencil,
        second: tuple[Stencil, ...], append_second: Stencil

    ) -> tuple[tuple[Stencil, ...], tuple[Stencil, ...]]:
        if len(first) == len(second):
            return first, second
        len_diff = len(first) - len(second)
        if len_diff > 0:
            new_second = list(second)
            for _ in range(len_diff):
                new_second.append(append_second.copy())
            second = tuple(new_second)
        else:
            new_first = list(first)
            for _ in range(-len_diff):
                new_first.append(append_first.copy())
            first = tuple(new_first)
        return first, second

    def _combine(
        self, other: "AxStencil", binary_op: Callable[[Stencil, Stencil], Stencil]
    ) -> "AxStencil":
        interior = binary_op(self.interior, other.interior)
        self_lefts, other_lefts = AxStencil.norm_offset(
            self.lefts, self.interior, other.lefts, other.interior
        )
        self_rights, other_rights = AxStencil.norm_offset(
            self.rights, self.interior, other.rights, other.interior
        )
        
        lefts = tuple(
            binary_op(self_lefts[i], other_lefts[i])
            for i in range(len(self_lefts))
        )
        rights = tuple(
            binary_op(self_rights[i], other_rights[i])
            for i in range(len(self_rights))
        )
        return AxStencil(interior, lefts, rights)

    def _scale(self, other: float) -> "AxStencil":
        interior = self.interior * other
        lefts = tuple(other * left for left in self.lefts)
        rights = tuple(other * right for right in self.rights)
        return AxStencil(interior, lefts, rights)

    def __neg__(self) -> "AxStencil":
        return self._scale(-1.0)

    def __add__(self, other) -> "AxStencil":
        if isinstance(other, AxStencil):
            return self._combine(other, lambda a, b: a + b)
        return NotImplemented

    def __sub__(self, other) -> "AxStencil":
        if isinstance(other, AxStencil):
            return self._combine(other, lambda a, b: a - b)
        return NotImplemented

    def __mul__(self, other) -> "AxStencil":
        if isinstance(other, AxStencil):
            return self._combine(other, lambda a, b: a * b)
        if isinstance(other, (float, int)):
            return self._scale(other)
        return NotImplemented

    def __rmul__(self, other) -> "AxStencil":
        if isinstance(other, (float, int)):
            return self._scale(other)
        return NotImplemented

    def __truediv__(self, other) -> "AxStencil":
        if isinstance(other, AxStencil):
            return self._combine(other, lambda a, b: a / b)
        if isinstance(other, (float, int)):
            return self._scale(1.0 / other)
        return NotImplemented
