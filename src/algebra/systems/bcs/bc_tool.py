from abc import ABC, abstractmethod
import numpy as np

from algebra.operator import Operator
from .bcs import BoundaryCondition

class BCTool(ABC):
    @abstractmethod
    def apply(self, bc: BoundaryCondition, operator: Operator):
        pass

    @abstractmethod
    def post_solve(self, bc: BoundaryCondition, field: np.ndarray):
        pass
