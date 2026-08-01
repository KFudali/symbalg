from abc import ABC, abstractmethod
from typing import Callable, Union
import numpy as np
from sparse import SparseArray

from algebra.space import shapes, Space

Array = Union[np.ndarray, SparseArray]


class Expression(ABC, shapes.SpaceShaped):
    @abstractmethod
    def eval(self) -> Array:
        pass


class FieldExpression(Expression):
    @abstractmethod
    def eval(self) -> np.ndarray:
        pass


class SparseExpression(Expression):
    @abstractmethod
    def eval(self) -> SparseArray:
        pass


class ConstFieldExpression(FieldExpression):
    def __init__(self, space: Space, value: np.ndarray | float):
        if isinstance(value, np.ndarray):
            super().__init__(shapes.Shape.from_array(space, value.shape))
            self._value = value
        else:
            super().__init__(shapes.Shape.scalar(space))
            self._value = np.array(value)

    def eval(self) -> np.ndarray:
        return self._value


class ConstSparseExpression(SparseExpression):
    def __init__(self, space: Space, value: SparseArray):
        super().__init__(shapes.Shape.from_sparse(space, value.shape))
        self._value = value

    def eval(self) -> SparseArray:
        return self._value


class CallableExpression(Expression):
    def __init__(self, shape: shapes.Shape, getter: Callable[[], Array]):
        super().__init__(shape)
        self._getter = getter

    def eval(self) -> Array:
        return self._getter()
