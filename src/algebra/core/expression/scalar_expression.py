from __future__ import annotations
from typing import Self, Callable
import numpy as np

from .expression import Expression


class ScalarExpression(Expression):
    def __init__(self, value: float | Callable[[], float]):
        super().__init__(())
        self._value = value

    def copy(self) -> Self:
        return ScalarExpression(self._value)

    def eval(self) -> np.ndarray:
        if isinstance(self._value, (float, int, np.floating, np.ndarray)):
            return np.array(self._value)
        return np.array(self._value())
