from abc import ABC, abstractmethod
from typing import Self

from algebra.space import Space
import scipy.sparse as sp
import numpy as np


class SparseExpression(ABC):
    def __init__(self, m: int, n: int):
        self._m = m
        self._n = n

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n, self.n)

    @property
    def m(self) -> int:
        return self._m

    @property
    def n(self) -> int:
        return self._n

    def compatible(self, space: Space) -> bool:
        n = np.prod(space.shape)
        return self.m == n and self.n == n

    @abstractmethod
    def copy(self) -> Self:
        pass

    @abstractmethod
    def eval(self) -> sp.spmatrix:
        pass


class ConstSparseExpression(SparseExpression):
    def __init__(self, mat: sp.spmatrix):
        super().__init__(mat.shape[0], mat.shape[1])
        self._mat = mat

    def copy(self) -> Self:
        return self.__class__(self._mat.copy())

    def eval(self) -> sp.spmatrix:
        return self._mat
