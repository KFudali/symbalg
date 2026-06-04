from typing import Generic
from abc import ABC, abstractmethod
import numpy as np

from algebra.operator import TOperator
from discrete.core.domain.boundary import TBoundary
from .bcs import BoundaryCondition


class BCTool(ABC, Generic[TOperator, TBoundary]):
    @abstractmethod
    def apply(
        self, bcs: list[BoundaryCondition[TBoundary]], lhs: TOperator, rhs: np.ndarray
    ) -> TOperator:
        pass

    @abstractmethod
    def post_solve(self, bcs: list[BoundaryCondition[TBoundary]], field: np.ndarray):
        pass
