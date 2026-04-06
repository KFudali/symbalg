from typing import Callable, Self
import copy
import numpy as np

from .expression import Expression
from algebra.exceptions import ShapeMismatchError

class CallableExpression(Expression):
    def __init__(
        self, 
        callable: Callable[[], np.ndarray], 
        output_shape: tuple[int, ...]
    ):
        super().__init__(output_shape)
        self._callable = callable

    def copy(self) -> Self:
        return CallableExpression(copy.deepcopy(self._callable), self.output_shape)

    def eval(self) -> np.ndarray:
        out = self._callable()
        if out.shape != self.output_shape:
            raise ShapeMismatchError((
                "CallableExpression has been constructed with callable that returns",
                f"array of shape {out.shape} that mismatches declared ",
                f"output_shape {self.output_shape}."
            ))
        return out