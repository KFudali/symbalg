from typing import Callable
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from tools import region

@dataclass(frozen=True)
class Stencil:
    _weights: dict[int, float]

    def copy(self) -> "Stencil":
        return Stencil(self._weights.copy())

    @property
    def weights(self) -> MappingProxyType:
        return MappingProxyType(self._weights)

    def offsets(self) -> list[int]:
        return list(self._weights.keys())

    def get(self, idx: int) -> float | None:
        return self._weights.get(idx)

    def __getitem__(self, idx: int) -> float:
        return self._weights[idx]

    def left_offsets(self) -> list[int]:
        return [abs(p) for p in self.offsets() if p < 0]

    def right_offsets(self) -> list[int]:
        return [p for p in self.offsets() if p > 0]

    def max_offsets(self) -> tuple[int, int]:
        left = 0
        right = 0
        if self.left_offsets():
            left = max(self.left_offsets())
        if self.right_offsets():
            right = max(self.right_offsets())
        return left, right

    def eval(self, ax: int, field: np.ndarray) -> np.ndarray:
        out_shape = list(field.shape)
        out_shape[ax] -= sum(self.max_offsets())
        out = np.zeros(shape=tuple(out_shape), dtype=field.dtype)
        self.eval_to(ax, field, out)
        return out

    def eval_to(self, ax: int, field: np.ndarray, out: np.ndarray):
        assert (field.shape[ax] - out.shape[ax]) == sum(self.max_offsets())
        offsets = [(0, 0) for _ in range(field.ndim)]
        offsets[ax] = self.max_offsets()
        interior = region.interior(field.ndim, tuple(offsets))
        for offset, weight in self.weights.items():
            offset_region = interior.shift(ax, offset)
            out += weight * field[offset_region]

    def _combine(
        self, other: "Stencil", binary_op: Callable[[float, float], float]
    ) -> "Stencil":
        weights = self._weights.copy()
        weights.update(other.weights)
        for offset, weight in self._weights.items():
            if offset in other.weights:
                weights[offset] = binary_op(weight, other.weights[offset])
        return Stencil(weights)

    def _combine_exclusive(
        self, other: "Stencil", binary_op: Callable[[float, float], float]
    ):
        shared = {}
        missing = {}
        for off in self._weights:
            if off in other.weights:
                shared[off] = binary_op(self._weights[off], other.weights[off])
            else:
                missing[off] = 0.0
        for off in other.weights:
            if off not in shared:
                missing[off] = 0.0
        shared.update(missing)
        return Stencil(shared)

    def _scale(self, scale: int | float) -> "Stencil":
        weights = self._weights.copy()
        for offset, weight in self._weights.items():
            weights[offset] = float(scale) * weight
        return Stencil(weights)

    def __neg__(self) -> "Stencil":
        return self._scale(-1.0)

    def __add__(self, other) -> "Stencil":
        if not isinstance(other, Stencil):
            return NotImplemented
        return self._combine(other, lambda a, b: a + b)

    def __sub__(self, other) -> "Stencil":
        if not isinstance(other, Stencil):
            return NotImplemented
        return self._combine(other, lambda a, b: a - b)

    def __mul__(self, other) -> "Stencil":
        if isinstance(other, Stencil):
            return self._combine_exclusive(other, lambda a, b: a * b)
        if isinstance(other, (float, int)):
            return self._scale(other)
        return NotImplemented

    def __rmul__(self, other) -> "Stencil":
        if isinstance(other, (float, int)):
            return self.__mul__(other)
        return NotImplemented

    def __truediv__(self, other) -> "Stencil":
        if isinstance(other, Stencil):
            if len(self._weights) != len(other._weights):
                raise ValueError("Can only div stencils of same offstes")
            for off, weight in other.weights.items():
                if off not in self._weights:
                    raise ValueError("Can only div stencils of same offstes")
                if np.isclose(weight, 0.0):
                    raise ValueError("Trying to devide using stencil with 0.0 weight")
            return self._combine(other, lambda a, b: a / b)
        if isinstance(other, (float, int)):
            return self._scale(1.0 / other)
        return NotImplemented
