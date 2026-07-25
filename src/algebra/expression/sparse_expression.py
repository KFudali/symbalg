from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
from algebra.space import FieldShape, FieldShaped, Space


class SparseExpression(ABC, FieldShaped):
    @abstractmethod
    def eval(self) -> sp.spmatrix:
        pass


class ConstSparseExpression(SparseExpression):
    def __init__(self, space: Space, value: sp.spmatrix | np.ndarray | float):
        if isinstance(value, sp.spmatrix):
            super().__init__(FieldShape.from_shape(space, value.shape))
            self._value = value
        elif isinstance(value, np.ndarray):
            super().__init__(FieldShape.from_shape(space, value.shape))
            self._value = sp.csr_matrix(value)
        else:
            super().__init__(FieldShape.scalar(space))
            self._value = sp.csr_matrix(np.array(value))
