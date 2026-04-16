from typing import Self
import numpy as np

from algebra.symbolic import SymbolicOperator
from algebra import Operator

class MockOperator(Operator):
    def __init__(self, name: str):
        super().__init__((10,10), (10,10))
        self.name = name
    def copy(self) -> Self: pass

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray): 
        pass

    def __neg__(self) -> Self: 
        return MockOperator(f"[-{self.name}]")

    def add(self, other: Self) -> Self:
        return MockOperator(f"[{self.name} + {other.name}]")

    def mul(self, other: Self) -> Self:
        return MockOperator(f"[{self.name} * {other.name}]")

    def scale(self, other: float) -> Self:
        return MockOperator(f"[{self.name} * {other} ]")

    def scale_arr(self, other: np.ndarray) -> Self:
        return MockOperator(f"[{self.name} * {other} ]")

def test_symbolic_operator():
    op_a = MockOperator("A")
    op_b = MockOperator("B")
    op_c = MockOperator("C")
    symbolic = SymbolicOperator[MockOperator](op_a)
    result = ((symbolic + op_b) * op_c).fold()
    assert result.name == "[[A + B] * C]"
