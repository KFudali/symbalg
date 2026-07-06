from typing import Self
import numpy as np

from tools.symbolic import BINARY_OPS, BinaryOpType
from algebra.expression import Expression
from algebra.operator import Operator
from algebra.space import ShapeTransform, Space

from .symbolic_expression import SymbolicExpression


class ArrayOperator(Operator):
    def __init__(self, space: Space, shape_transform: ShapeTransform, expr: Expression):
        super().__init__(space, shape_transform)
        self._expr = SymbolicExpression.wrap(expr)

    @property
    def expression(self) -> SymbolicExpression:
        return self._expr

    def copy(self) -> Self:
        return self.__class__(self.space, self.shape_transform, self._expr.copy())

    def _apply(self, ax: int, inp: np.ndarray, out: np.ndarray):
        out[:] += self._expr.eval() * inp

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        if isinstance(other, ArrayOperator):
            return self.__class__(
                self.space,
                self.shape_transform,
                BINARY_OPS[optype](self._expr, other.expression),
            )
        return NotImplemented

    def _scale(self, other: float) -> Self:
        return self.__class__(self.space, self.shape_transform, other * self._expr)

    def __neg__(self) -> Self:
        return self.__class__(self.space, self.shape_transform, -self._expr)
