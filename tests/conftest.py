from typing import Self, Callable
import numpy as np

from tools.symbolic.optype import BinaryOpType
from algebra.operator import Operator
from algebra.space import Space, ShapeTransform


class MockOperator(Operator):
    def __init__(self, name: str):
        space = Space((10, 10))
        super().__init__(space, ShapeTransform.NONE)
        self.name = name
        self._apply_callable = None

    def copy(self) -> Self:
        return self.__class__(self.name)

    def set_apply(self, apply: Callable[[np.ndarray, np.ndarray], None]):
        self._apply_callable = apply

    def apply(self, inp: np.ndarray, out: np.ndarray):
        if self._apply_callable:
            self._apply_callable(inp, out)

    def _combine(self, other: Self, optype: BinaryOpType) -> Self:
        if optype == BinaryOpType.ADD:
            return self.add(other)
        if optype == BinaryOpType.SUB:
            return self.sub(other)
        if optype == BinaryOpType.DIV:
            return self.div(other)
        if optype == BinaryOpType.MUL:
            return self.mul(other)

    def _scale(self, other: float) -> Self:
        return self.__class__(f"[{self.name} * {other}]")

    def __neg__(self) -> Self:
        return self.__class__(f"[-{self.name}]")

    def add(self, other: Self) -> Self:
        return self.__class__(f"[{self.name} + {other.name}]")

    def sub(self, other: Self) -> Self:
        return self.__class__(f"[{self.name} - {other.name}]")

    def mul(self, other: Self) -> Self:
        return self.__class__(f"[{self.name} * {other.name}]")

    def div(self, other: Self) -> Self:
        return self.__class__(f"[{self.name} / {other.name}]")
