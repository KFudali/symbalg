from __future__ import annotations
from typing import Callable, Self
import copy
import numpy as np

from algebra.exceptions import ShapeMismatchError
from algebra.fieldshape import FieldShape
from .expression import Expression


class CallableExpression(Expression):
    def __init__(self, getter: Callable[[], np.ndarray], shape: FieldShape):
        super().__init__(shape)
        self._getter = getter

    def copy(self) -> "CallableExpression":
        return CallableExpression(copy.deepcopy(self._getter), self.shape)

    def eval(self) -> np.ndarray:
        out = self._getter()
        if out.shape != self.shape:
            raise ShapeMismatchError(
                (
                    "CallableExpression has been constructed with callable that returns",
                    f"array of shape {out.shape} that mismatches declared ",
                    f"output_shape {self.shape}.",
                )
            )
        return out


class ZeroExpression(CallableExpression):
    def __init__(self, shape: FieldShape):
        def zeros():
            return np.zeros(shape=shape, dtype=float)

        super().__init__(zeros, shape)
