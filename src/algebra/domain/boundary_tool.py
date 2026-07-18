from abc import ABC, abstractmethod
import numpy as np

from algebra.operator import Operator
from .bcs import BoundaryCondition


class BoundaryTool(ABC):
    @abstractmethod
    def apply_bcs(
        self, bcs: BoundaryCondition, operator: Operator, rhs: np.ndarray
    ) -> Operator:
        pass

    @abstractmethod
    def normalize(self, bcs: BoundaryCondition, field: np.ndarray):
        pass
