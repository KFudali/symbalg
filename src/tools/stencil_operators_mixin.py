from __future__ import annotations

import numbers
from typing import Dict, Protocol, TypeVar, Self

class StencilLike(Protocol):
    contribs: Dict[int, Dict[int, float]]

    def copy(self) -> Self: ...
S = TypeVar("S", bound=StencilLike)

class StencilOperatorsMixin:
    """
    Algebra operations for stencil-like objects.

    Requires:
        - self.contribs: dict[int, dict[int, float]]
        - self.copy() -> Self
    """

    def _combine(self: S, other: S, op) -> S:
        if not isinstance(other, self.__class__):
            return NotImplemented

        result = self.copy()

        for ax, offsets in other.contribs.items():
            if ax not in result.contribs:
                result.contribs[ax] = {}

            for k, c in offsets.items():
                result.contribs[ax][k] = op(
                    result.contribs[ax].get(k, 0.0),
                    c,
                )

        return result

    def __add__(self: S, other: S) -> S:
        return self._combine(other, lambda a, b: a + b)

    def __sub__(self: S, other: S) -> S:
        return self._combine(other, lambda a, b: a - b)

    def __neg__(self: S) -> S:
        result = self.copy()
        for ax in result.contribs:
            for k in result.contribs[ax]:
                result.contribs[ax][k] *= -1.0
        return result

    def _scale(self: S, scalar: float) -> S:
        result = self.copy()
        for ax in result.contribs:
            for k in result.contribs[ax]:
                result.contribs[ax][k] *= scalar
        return result

    def __mul__(self: S, other: float | S) -> S:
        if isinstance(other, numbers.Number):
            return self._scale(float(other))

        if isinstance(other, self.__class__):
            result = self.__class__({})

            for ax1, offsets1 in self.contribs.items():
                for ax2, offsets2 in other.contribs.items():
                    if ax1 != ax2:
                        raise NotImplementedError(
                            "Stencil multiplication only supported along same axis."
                        )

                    if ax1 not in result.contribs:
                        result.contribs[ax1] = {}

                    for k1, c1 in offsets1.items():
                        for k2, c2 in offsets2.items():
                            k = k1 + k2
                            result.contribs[ax1][k] = (
                                result.contribs[ax1].get(k, 0.0)
                                + c1 * c2
                            )

            return result

        return NotImplemented

    def __rmul__(self: S, other: float) -> S:
        return self.__mul__(other)

    def __truediv__(self: S, scalar: float) -> S:
        return self._scale(1.0 / float(scalar))
