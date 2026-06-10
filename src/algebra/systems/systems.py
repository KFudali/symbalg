from dataclasses import dataclass
import numpy as np
from algebra.operator import Operator


@dataclass(frozen=True)
class LinearSystem:
    lhs: Operator
    rhs: np.ndarray

    def copy(self) -> "LinearSystem":
        return LinearSystem(self.lhs.copy(), self.rhs.copy())
