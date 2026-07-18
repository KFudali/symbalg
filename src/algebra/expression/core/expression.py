from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from algebra.space import FieldShape, FieldShaped, Space


class Expression(ABC, FieldShaped):
    @abstractmethod
    def eval(self) -> np.ndarray:
        pass


class ConstExpression(Expression):
    def __init__(self, space: Space, value: np.ndarray | float):
        if isinstance(value, np.ndarray):
            super().__init__(FieldShape.from_shape(space, value.shape))
            self._value = value
        else:
            super().__init__(FieldShape.scalar(space))
            self._value = np.array(value)

    def eval(self) -> np.ndarray:
        return self._value


class CallableExpression(Expression):
    def __init__(self, shape: FieldShape, getter: Callable[[], np.ndarray]):
        super().__init__(shape)
        self._getter = getter

    def eval(self) -> np.ndarray:
        return self._getter()
