from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Self
import scipy.sparse as sp
import numpy as np


from algebra.space import Space, SparseShape, SparseShaped


class SparseExpression(ABC, SparseShaped):
    def compatible(self, space: Space) -> bool:
        n = int(np.prod(space.shape))
        return self.shape[-1] == n and self.shape[-2] == n

    @abstractmethod
    def copy(self) -> Self:
        pass

    @abstractmethod
    def eval(self) -> sp.sparray:
        pass


class ConstSparseExpression(SparseExpression):
    def __init__(self, space: Space, array: sp.sparray):
        super().__init__(SparseShape.from_shape(space, array.shape))
        self._array = array

    def compatible(self, space: Space) -> bool:
        n = int(np.prod(space.shape))
        return self._array.shape[-2] == n and self._array.shape[-1] == n

    def copy(self) -> Self:
        return self.__class__(self.space, self._array.copy())

    def eval(self) -> sp.sparray:
        return self._array


class CallableSparseExpression(SparseExpression):
    def __init__(
        self,
        shape: SparseShape,
        getter: Callable[[], sp.sparray],
    ):
        super().__init__(shape)
        self._getter = getter

    def copy(self) -> Self:
        return self.__class__(self.sparseshape, self._getter)

    def eval(self) -> sp.sparray:
        return self._getter()
