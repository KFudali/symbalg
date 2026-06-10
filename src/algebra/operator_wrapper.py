from __future__ import annotations

from typing import Callable, Self
import numpy as np

from tools.symbolic.optype import BinaryOpType

from .operator import Operator


ApplyHook = Callable[[np.ndarray, np.ndarray], None]


class OperatorWrapper(Operator):
    """Wraps an :class:`Operator` and augments its ``apply`` with an extra hook.

    All other behaviour (space, shape transform, algebraic magics, ``copy``,
    ``of`` …) is delegated to the wrapped operator, so the wrapper is a
    drop-in replacement. Algebraic combinations (``+``, ``-``, ``*`` …) and
    ``copy`` re-wrap the resulting operator with the same hook so the hook
    keeps being applied after the wrapper participates in expressions.
    """

    def __init__(self, inner: Operator, hook: ApplyHook):
        super().__init__(inner.space, inner.shape_transform)
        self._inner = inner
        self._hook = hook

    @property
    def inner(self) -> Operator:
        return self._inner

    @property
    def hook(self) -> ApplyHook:
        return self._hook

    def _wrap(self, op: Operator) -> "OperatorWrapper":
        return OperatorWrapper(op, self._hook)

    def apply(self, inp: np.ndarray, out: np.ndarray) -> None:
        self._inner.apply(inp, out)
        self._hook(inp, out)

    def copy(self) -> Self:
        return self._wrap(self._inner.copy())

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        other_inner = other.inner if isinstance(other, OperatorWrapper) else other
        return self._wrap(self._inner._combine(other_inner, optype))

    def _scale(self, other: float) -> Self:
        return self._wrap(self._inner._scale(other))

    def __neg__(self) -> Self:
        return self._wrap(-self._inner)

    def __getattr__(self, name: str):
        # Called only when normal attribute lookup fails — delegates everything
        # else (e.g. discretization-specific helpers) to the wrapped operator.
        return getattr(self._inner, name)
