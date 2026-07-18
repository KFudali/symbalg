from __future__ import annotations

from dataclasses import dataclass
from typing import Generic
import numpy as np
from algebra.operator import TOperator
from algebra.expression import Expression


@dataclass(frozen=True)
class LinearSystem(Generic[TOperator]):
    lhs: TOperator
    rhs: np.ndarray

    def copy(self) -> "LinearSystem":
        return LinearSystem(self.lhs.copy(), self.rhs.copy())
