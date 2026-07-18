from __future__ import annotations

from typing import Callable, Self
import numpy as np

from tools.symbolic.optype import BinaryOpType
from algebra.operator import Operator

ApplyHook = Callable[[np.ndarray, np.ndarray], None]


class OperatorWrapper(Operator):
    def __init__(self, operator: Operator):
        super().__init__(operator.space, operator.shape_transform)
        self._operator = operator

    def apply(self, inp: np.ndarray, out: np.ndarray) -> None:
        return self._operator.apply(inp, out)

    def copy(self) -> Self:
        return self.__class__(self._operator)

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        return self.__class__(self._operator._combine(other, optype))

    def _scale(self, other: float) -> Self:
        return self.__class__(self._operator._scale(other))

    def __neg__(self) -> Self:
        return self.__class__(self._operator.__neg__())


class ApplyHookOperator(OperatorWrapper):
    """Wraps an :class:`Operator` and augments its ``apply`` with an extra hook.

    All other behaviour (space, shape transform, algebraic magics, ``copy``,
    ``of`` …) is delegated to the wrapped operator, so the wrapper is a
    drop-in replacement. Algebraic combinations (``+``, ``-``, ``*`` …) and
    ``copy`` re-wrap the resulting operator with the same hook so the hook
    keeps being applied after the wrapper participates in expressions.
    """

    def __init__(self, operator: Operator, hook: ApplyHook):
        super().__init__(operator)
        self._hook = hook

    @property
    def hook(self) -> ApplyHook:
        return self._hook

    def apply(self, inp: np.ndarray, out: np.ndarray) -> None:
        self._operator.apply(inp, out)
        self._hook(inp, out)
