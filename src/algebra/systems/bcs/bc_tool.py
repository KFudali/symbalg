from abc import ABC, abstractmethod
import numpy as np

from algebra.core.operator import Operator
from .bcs import BoundaryCondition


class BCTool(ABC):
    @abstractmethod
    def apply(self, bcs: list[BoundaryCondition], lhs: Operator, rhs: np.ndarray):
        pass

    @abstractmethod
    def post_solve(self, bc: BoundaryCondition, field: np.ndarray):
        pass
