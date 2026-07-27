from abc import ABC, abstractmethod
from typing import Callable, Self

from algebra.space import Space, SparseShape, SparseShaped
import scipy.sparse as sp
import numpy as np


class SparseExpression(ABC, SparseShaped):
    def __init__(self, shape: SparseShape):
        super().__init__(shape)

    def compatible(self, space: Space) -> bool:
        n = int(np.prod(space.shape))
        return self.shape[-1] == n and self.shape[-2] == n

    @abstractmethod
    def copy(self) -> Self:
        pass

    @abstractmethod
    def eval(self) -> sp.spmatrix:
        pass


class ConstSparseExpression(SparseExpression):
    def __init__(self, space: Space, mat: sp.spmatrix):
        super().__init__(SparseShape(space, ()))
        self._mat = mat

    def compatible(self, space: Space) -> bool:
        n = int(np.prod(space.shape))
        return self._mat.shape == (n, n)

    def copy(self) -> Self:
        return self.__class__(self.space, self._mat.copy())

    def eval(self) -> sp.spmatrix:
        return self._mat


class CallableSparseExpression(SparseExpression):
    def __init__(self, space: Space, getter: Callable[[], sp.spmatrix]):
        super().__init__(SparseShape(space, ()))
        self._getter = getter

    def copy(self) -> Self:
        return self.__class__(self.space, self._getter)

    def eval(self) -> sp.spmatrix:
        return self._getter()
