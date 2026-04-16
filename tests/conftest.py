from typing import Self, Callable
import numpy as np
from algebra.core.operator import Operator

class MockOperator(Operator):
    def __init__(self, name: str, input_shape = (10,), output_shape = (10,)):
        super().__init__(input_shape, output_shape)
        self.name = name
        self._apply_callable = None

    def copy(self) -> Self:
        pass

    def set_apply(self, apply: Callable[[np.ndarray, np.ndarray], None]):
        self._apply_callable = apply

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        if self._apply_callable:
            self._apply_callable(input_field, output_field)

    def __neg__(self) -> Self: 
        return MockOperator(f"[-{self.name}]")

    def add(self, other: Self) -> Self:
        return MockOperator(f"[{self.name} + {other.name}]")

    def sub(self, other: Self) -> Self:
        return MockOperator(f"[{self.name} - {other.name}]")

    def mul(self, other: Self) -> Self:
        return MockOperator(f"[{self.name} * {other.name}]")

    def div(self, other: Self) -> Self:
        return MockOperator(f"[{self.name} / {other.name}]")

    def scale(self, other: float) -> Self:
        return MockOperator(f"[{self.name} * {other}]")

    def scale_arr(self, other: np.ndarray) -> Self:
        return MockOperator(f"[{self.name} * {other}]")
