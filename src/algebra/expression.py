from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from .exceptions import ShapeMismatchError


class Expression(ABC):
    def __init__(self, shape: tuple[int, ...]):
        self._shape = shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @abstractmethod
    def eval(self) -> np.ndarray:
        pass


class CallableExpression(Expression):
    def __init__(self, shape: tuple[int, ...], getter: Callable[[], np.ndarray]):
        super().__init__(shape)
        self._getter = getter

    def eval(self) -> np.ndarray:
        out = self._getter()
        if out.shape != self.shape:
            raise ShapeMismatchError(
                (
                    "CallableExpression was created with getter that does not match",
                    f"declared FieldShape {self.shape}. getter shape: {out.shape}.",
                )
            )
        return out


class ScalarExpression(Expression):
    def __init__(self, scalar: float):
        super().__init__(())
        self._scalar = np.array(scalar)

    def eval(self) -> np.ndarray:
        return self._scalar


class CallableScalarExpression(Expression):
    def __init__(self, scalar_getter: Callable[[], float]):
        super().__init__(())
        self._scalar_getter = scalar_getter

    def eval(self) -> np.ndarray:
        return np.array(self._scalar_getter())
