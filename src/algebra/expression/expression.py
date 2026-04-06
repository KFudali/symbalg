from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Self
import numpy as np


class Expression(ABC):
    def __init__(self, output_shape: tuple[int, ...]):
        self._output_shape = output_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    @abstractmethod
    def copy(self) -> Self:
        pass

    @abstractmethod
    def eval(self) -> np.ndarray:
        pass
