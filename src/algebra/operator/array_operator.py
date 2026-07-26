from typing import Self
import numpy as np

from tools.symbolic import BinaryOpType, BINARY_OPS
from algebra.space import Space, ShapeTransform
from algebra.expression import SparseExpression
from algebra.expression.symbolic import SymbolicSparseExpression
from algebra.exceptions import ShapeMismatchError
from .core import Operator


class ArrayOperator(Operator):
    def __init__(
        self,
        space: Space,
        shape_transform: ShapeTransform,
        mat: SparseExpression,
    ):
        if not mat.compatible(space):
            raise ShapeMismatchError(
                f"sparse expr{mat} not compatible with space: {space}."
            )
        super().__init__(space, shape_transform)
        self._mat = SymbolicSparseExpression.wrap(mat)

    @property
    def mat(self) -> SymbolicSparseExpression:
        return self._mat

    def copy(self) -> Self:
        return self.__class__(self.space, self._shape_transform, self._mat.copy())

    def _apply(self, ax: int, inp: np.ndarray, out: np.ndarray):
        if ax > 0 and self._shape_transform is ShapeTransform.NONE:
            return
        mat = self._mat.eval()
        out[:] += (mat @ inp.ravel()).reshape(out.shape)

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        op = BINARY_OPS[optype]
        if not isinstance(other, ArrayOperator):
            return NotImplemented
        combined_mat = op(self.mat, other.mat)
        return self.__class__(self.space, self.shape_transform, combined_mat)

    def _scale(self, other: float) -> Self:
        return self.__class__(self.space, self.shape_transform, other * self.mat)
