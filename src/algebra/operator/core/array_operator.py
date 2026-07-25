from typing import Self
import numpy as np
import scipy.sparse as sp

from tools.symbolic import BinaryOpType
from algebra.space import ShapeTransform, Space
from .operator import Operator


class ArrayOperator(Operator):
    def __init__(
        self, space: Space, shape_transform: ShapeTransform, matrix: sp.spmatrix
    ):
        super().__init__(space, shape_transform)
        self._matrix = matrix

    @property
    def matrix(self) -> sp.spmatrix:
        return self._matrix

    def copy(self) -> Self:
        return self.__class__(self.space, self.shape_transform, self._matrix.copy())

    def apply(self, inp: np.ndarray, out: np.ndarray):
        field_rank = inp.ndim - self._space.ndim
        if field_rank == 0:
            out[:] = (self._matrix @ inp.ravel()).reshape(out.shape)
        else:
            for comp in range(inp.shape[0]):
                self.apply(inp[comp], out[comp])

    def _apply(self, ax: int, inp: np.ndarray, out: np.ndarray):
        pass

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        if isinstance(other, ArrayOperator):
            if optype == BinaryOpType.ADD:
                return self.__class__(
                    self.space, self.shape_transform, self.matrix + other.matrix
                )
            if optype == BinaryOpType.SUB:
                return self.__class__(
                    self.space, self.shape_transform, self.matrix - other.matrix
                )
            return NotImplemented
        return NotImplemented

    def _scale(self, other: float) -> Self:
        return self.__class__(self.space, self.shape_transform, self.matrix * other)
