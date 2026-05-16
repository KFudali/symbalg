from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Self, TypeVar
import numpy as np
from algebra.fieldshape import FieldShape, FieldShaped


class Expression(FieldShaped, ABC):
    def __init__(self, shape: FieldShape):
        super().__init__(shape)

    @abstractmethod
    def copy(self) -> Self:
        pass

    @abstractmethod
    def eval(self) -> np.ndarray:
        pass


TExpression = TypeVar("TExpression", bound=Expression)
